"""Post-hoc frozen KLUE-RoBERTa feature probe; no backbone fine-tuning.

V2 evaluation has been inspected. These are exploratory paired comparisons,
not a new held-out confirmation. Input features never use gold span roles.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from model import amounts
from run import ROOT, read, select_threshold, summarize


def normalized(text):
    """Currency offsets come from a parser, never from annotated roles."""
    out, spans, cursor = "", [], 0

    def clean(s):
        s = re.sub(r"\d+", "숫자", s)
        return re.sub(r"[TVE]숫자가상사업", "가상사업", s)

    for span in amounts(text):
        out += clean(text[cursor : span["start"]])
        start = len(out)
        out += "금액"
        spans.append((start, len(out)))
        cursor = span["end"]
    out += clean(text[cursor:])
    return out, spans


def features(texts, encoder_path, out):
    torch.set_num_threads(4)
    torch.manual_seed(731)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    encoder = (
        AutoModel.from_pretrained(
            encoder_path, local_files_only=True, add_pooling_layer=False
        )
        .cpu()
        .eval()
    )
    unique = sorted(set(texts))
    features_by_text = {}
    started = time.monotonic()
    max_tokens = 0
    for first in range(0, len(unique), 8):
        batch = unique[first : first + 8]
        tokenized = tokenizer(
            batch,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = tokenized.pop("offset_mapping").numpy()
        max_tokens = max(max_tokens, int(tokenized["attention_mask"].sum(1).max()))
        assert tokenized["input_ids"].shape[1] <= 512, "No silent truncation"
        with torch.inference_mode():
            hidden = encoder(**tokenized).last_hidden_state.numpy()
        for i, text in enumerate(batch):
            valid = offsets[i, :, 1] > offsets[i, :, 0]
            mean = hidden[i, valid].mean(0)
            relation = np.concatenate([hidden[i, 0], mean])
            spans = [(m.start(), m.end()) for m in re.finditer("금액", text)]
            # The caller selects parser-generated offsets; ordinary lexical
            # occurrences of '금액' must not become argument candidates.
            token_features = {}
            for start, end in spans:
                overlap = (offsets[i, :, 0] < end) & (offsets[i, :, 1] > start) & valid
                assert overlap.any()
                token_features[f"{start}:{end}"] = np.concatenate(
                    [hidden[i, overlap].mean(0), mean]
                )
            features_by_text[text] = (relation, token_features)
        print(
            json.dumps(
                {
                    "encoded": min(first + 8, len(unique)),
                    "unique": len(unique),
                    "seconds": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )
    record = {
        "source": "klue/roberta-base",
        "checkpoint_directory": Path(encoder_path).name,
        "unique_masked_passages": len(unique),
        "max_tokens": max_tokens,
        "truncated": 0,
        "seconds": round(time.monotonic() - started, 3),
        "torch": torch.__version__,
        "device": "CPU",
        "threads": 4,
        "weights_updated": False,
        "encoder_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(encoder_path).glob("*")
            if p.name in ["model.safetensors", "pytorch_model.bin", "config.json"]
        },
    }
    (out / "encoder_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return features_by_text


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs", random_state=731),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not (args.out / "results.json").exists()
    lock = json.loads((ROOT / "encoder_freeze.json").read_text())
    for name, digest in lock["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    args.out.mkdir(parents=True, exist_ok=True)
    datasets = {s: read(s) for s in ["train", "validation", "evaluation"]}
    official_path = ROOT / "data/official_cases.jsonl"
    if official_path.exists():
        datasets["official_cases"] = [
            json.loads(s) for s in official_path.read_text().splitlines()
        ]
    norm = {
        r["text"]: normalized(r["text"]) for rows in datasets.values() for r in rows
    }
    encoded = features([t for t, _ in norm.values()], args.encoder, args.out)

    def relation_features(rows):
        return np.stack([encoded[norm[r["text"]][0]][0] for r in rows])

    def span_features(row):
        text, spans = norm[row["text"]]
        return (
            np.stack([encoded[text][1][f"{a}:{b}"] for a, b in spans])
            if spans
            else np.empty((0, 1536))
        )

    train = datasets["train"]
    relation = classifier()
    relation.fit(relation_features(train), [r["relation"] for r in train])
    role = classifier()
    eligible = [r for r in train if r["relation"] != "undetermined"]
    role.fit(
        np.concatenate([span_features(r) for r in eligible]),
        [s["role"] for r in eligible for s in r["spans"]],
    )

    def predict(rows):
        probabilities = relation.predict_proba(relation_features(rows))
        output = []
        for r, p in zip(rows, probabilities):
            spans = amounts(r["text"])
            if spans:
                probs = role.predict_proba(span_features(r))
                for span, q in zip(spans, probs):
                    span.update(
                        role=str(role.classes_[q.argmax()]), confidence=float(q.max())
                    )
            output.append(
                {
                    "relation": str(relation.classes_[p.argmax()]),
                    "confidence": float(p.max()),
                    "relation_probabilities": dict(
                        zip(map(str, relation.classes_), map(float, p))
                    ),
                    "candidates": spans,
                }
            )
        return output

    dev_raw = predict(datasets["validation"])
    threshold, trace = select_threshold(datasets["validation"], dev_raw)
    (args.out / "selection.json").write_text(
        json.dumps({"threshold": threshold, "validation_candidates": trace}, indent=2)
        + "\n"
    )
    results = {
        "status": "post-hoc exploratory; v2 evaluation already inspected",
        "threshold": threshold,
        "sets": {},
    }
    for split in [s for s in datasets if s != "train"]:
        rows = datasets[split]
        raw = dev_raw if split == "validation" else predict(rows)
        methods, scores = summarize(rows, raw, threshold)
        (args.out / f"{split}_predictions.json").write_text(
            json.dumps(
                {"ids": [r["id"] for r in rows], "raw": raw, "methods": methods},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        results["sets"][split] = scores
    for name, pipeline in [("relation", relation), ("roles", role)]:
        scaler, clf = pipeline.steps[0][1], pipeline.steps[1][1]
        np.savez_compressed(
            args.out / f"{name}_probe.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
            coefficient=clf.coef_,
            intercept=clf.intercept_,
            classes=clf.classes_,
        )
    (args.out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
