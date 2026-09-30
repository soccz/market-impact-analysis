"""Fixed-protocol CPU study; outputs can be reproduced in a new directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from model import SpanModel, correct, decision, keyword, ordered, signature

ROOT = Path(__file__).resolve().parent
LABELS = ["correction", "policy_change", "equivalent", "undetermined"]


def read(split):
    return [json.loads(s) for s in (ROOT / f"data/{split}.jsonl").read_text().splitlines()]


def metrics(rows, predictions):
    hits = [correct(r, p) for r, p in zip(rows, predictions)]
    accepted = [i for i, p in enumerate(predictions) if p["accepted"]]
    truth = [r["relation"] for r in rows]
    pred = [p["relation"] for p in predictions]
    return {"n": len(rows), "relation_accuracy": accuracy_score(truth, pred),
            "macro_f1": f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0),
            "confusion": confusion_matrix(truth, pred, labels=LABELS).tolist(),
            "argument_exact": float(np.mean([signature(r["spans"]) == signature(p["arguments"]) for r, p in zip(rows, predictions)])),
            "joint_exact": float(np.mean(hits)), "accepted": len(accepted), "coverage": len(accepted) / len(rows),
            "accepted_errors": sum(not hits[i] for i in accepted),
            "selective_joint_error": sum(not hits[i] for i in accepted) / len(accepted) if accepted else None,
            "correctly_answered_fraction": sum(hits[i] for i in accepted) / len(rows)}


def select_threshold(rows, raw):
    candidates = []
    for threshold in [i / 20 for i in range(20)] + [.99]:
        m = metrics(rows, [decision(r, True, threshold) for r in raw])
        candidates.append({"threshold": threshold, **m})
    eligible = [r for r in candidates if r["accepted"] >= 20 and r["selective_joint_error"] <= .05]
    chosen = min(eligible, key=lambda r: (-r["coverage"], r["threshold"]))["threshold"] if eligible else 1.01
    return chosen, candidates


def summarize(rows, raw, threshold):
    methods = {
        "keyword_order": [decision(keyword(r["text"])) for r in rows],
        "learned_order": [decision(ordered(p)) for p in raw],
        "span_model": [decision(p) for p in raw],
        "constrained": [decision(p, True) for p in raw],
        "selective": [decision(p, True, threshold) for p in raw],
    }
    scores = {method: metrics(rows, ps) for method, ps in methods.items()}
    return methods, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not (args.out / "results.json").exists(), "Refusing to overwrite a completed study."
    lock = json.loads((ROOT / "freeze.json").read_text())
    for name, digest in lock["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = SpanModel()
    model.fit(read("train"))
    dev = read("validation")
    dev_raw = model.predict([r["text"] for r in dev])
    threshold, trace = select_threshold(dev, dev_raw)
    # Threshold selection completes before any evaluation prediction.
    (args.out / "selection.json").write_text(json.dumps({"threshold": threshold, "validation_candidates": trace}, indent=2) + "\n")
    result = {"threshold": threshold, "labels": LABELS, "sets": {}}
    for split in ["validation", "evaluation"]:
        rows = read(split)
        raw = dev_raw if split == "validation" else model.predict([r["text"] for r in rows])
        methods, scores = summarize(rows, raw, threshold)
        (args.out / f"{split}_predictions.json").write_text(json.dumps({"ids": [r["id"] for r in rows], "raw": raw, "methods": methods}, ensure_ascii=False, indent=2) + "\n")
        groups = sorted({r["frame"] for r in rows})
        per_frame = {}
        for method, ps in methods.items():
            hits = np.array([correct(r, p) for r, p in zip(rows, ps)])
            per_frame[method] = [float(hits[[r["frame"] == g for r in rows]].mean()) for g in groups]
            scores[method]["per_family"] = {str(f): metrics([r for r in rows if r["family"] == f], [p for r, p in zip(rows, ps) if r["family"] == f]) for f in sorted({r["family"] for r in rows})}
            semantic = sorted({r["semantic_id"] for r in rows})
            scores[method]["all_surface_families_correct"] = float(np.mean([all(hit for r, hit in zip(rows, hits) if r["semantic_id"] == g) for g in semantic]))
            scores[method]["per_subtype"] = {sub: metrics([r for r in rows if r["subtype"] == sub], [p for r, p in zip(rows, ps) if r["subtype"] == sub]) for sub in sorted({r["subtype"] for r in rows})}
        delta = np.array(per_frame["span_model"]) - np.array(per_frame["learned_order"])
        rng = np.random.default_rng(731)
        boot = delta[rng.integers(0, len(groups), size=(10000, len(groups)))].mean(axis=1)
        result["sets"][split] = {"metrics": scores, "span_minus_order_joint": {"delta": float(delta.mean()), "frame_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(), "groups": len(groups)}}
    # Store portable learned coefficients; no pickle execution is needed to inspect them.
    for name, pipeline in [("relation", model.relation), ("roles", model.roles)]:
        vectorizer, clf = pipeline.steps[0][1], pipeline.steps[1][1]
        (args.out / f"{name}_features.json").write_text(json.dumps({"vocabulary": vectorizer.vocabulary_, "classes": clf.classes_.tolist()}, ensure_ascii=False) + "\n")
        np.savez_compressed(args.out / f"{name}_weights.npz", idf=vectorizer.idf_, coefficients=clf.coef_, intercept=clf.intercept_)
    result["runtime"] = {"seconds": round(time.monotonic() - started, 3), "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "device": "CPU", "iterations": {n: m.steps[-1][1].n_iter_.tolist() for n, m in [("relation", model.relation), ("roles", model.roles)]}}
    (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"threshold": threshold, "evaluation": {k: {n: v for n, v in m.items() if n not in ["per_family", "per_subtype", "confusion"]} for k, m in result["sets"]["evaluation"]["metrics"].items()}, "runtime": result["runtime"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
