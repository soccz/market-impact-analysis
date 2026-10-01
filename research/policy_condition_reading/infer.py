"""Inference is label-blind: this module never opens reference.json."""

import argparse
import json
import time
from pathlib import Path

from prepare import load
from reader import Encoder, chunks, predict, tfidf_ranks, read_source, sha

ROOT = Path(__file__).resolve().parent


def write(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def run(cache, model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    docs = load(cache)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    weights = Path(model) / "model.safetensors"
    if sha(weights.read_bytes()) != protocol["model"]["weights_sha256"]:
        raise ValueError("Unexpected model weights")
    inputs = ["reader.py", "prepare.py", "infer.py", "protocol.json", "sources.json"]
    write(
        output / "inference_inputs.json",
        {name: sha((ROOT / name).read_bytes()) for name in inputs},
    )
    start = time.monotonic()
    encoder = Encoder(model)
    predictions = {}
    for doc, pages in docs.items():
        windows = chunks(pages)
        lexical = tfidf_ranks(windows)
        neural = encoder.ranks(windows)
        predictions[doc] = {
            "document_first": predict(pages, "document_first"),
            "tfidf_top1": predict(pages, "tfidf_top1", lexical),
            "klue_top1": predict(pages, "klue_top1", neural),
            "tfidf_consensus3": predict(pages, "tfidf_consensus3", lexical),
        }
        # Fixed ingestion ablation, same lexical method and unchanged parser.
        raw = read_source(Path(cache) / (doc + ".bin"), layout=False)
        predictions[doc]["unsorted_tfidf_top1"] = predict(
            raw, "tfidf_top1", tfidf_ranks(chunks(raw))
        )
        print(doc, len(windows), "windows", flush=True)
    write(output / "predictions.json", predictions)
    write(
        output / "runtime.json",
        dict(
            device="cpu",
            torch_threads=4,
            elapsed_seconds=round(time.monotonic() - start, 3),
            new_fits=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.cache, args.model, args.output)
