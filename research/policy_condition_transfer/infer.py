"""First application of the unchanged readers. No new reference labels loaded."""

import argparse
import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "policy_condition_reading"
sys.path.insert(0, str(PREVIOUS))
from reader import Encoder, chunks, predict, read_source, sha, tfidf_ranks
from refined import read as refined_read


def write(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def verify_freeze():
    for relative, expected in json.loads((ROOT / "freeze.json").read_text())[
        "files"
    ].items():
        assert sha((ROOT.parent / relative).read_bytes()) == expected, relative


def load(cache):
    docs = {}
    for source in json.loads((ROOT / "sources.json").read_text()):
        path = Path(cache) / (source["id"] + ".bin")
        assert sha(path.read_bytes()) == source["sha256"], source["id"]
        pages = read_source(path)
        assert (
            sha(json.dumps(pages, ensure_ascii=False).encode()) == source["text_sha256"]
        )
        docs[source["id"]] = pages
    return docs


def run(cache, model, output):
    verify_freeze()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    docs = load(cache)
    expected_model = json.loads((PREVIOUS / "protocol.json").read_text())["model"]
    assert (
        sha((Path(model) / "model.safetensors").read_bytes())
        == expected_model["weights_sha256"]
    )
    start = time.monotonic()
    encoder = Encoder(model)
    predictions = {}
    for doc, pages in docs.items():
        windows = chunks(pages)
        lexical, neural = tfidf_ranks(windows), encoder.ranks(windows)
        refined, region = refined_read(pages)
        predictions[doc] = {
            "document_first": predict(pages, "document_first"),
            "tfidf_top1": predict(pages, "tfidf_top1", lexical),
            "klue_top1": predict(pages, "klue_top1", neural),
            "tfidf_consensus3": predict(pages, "tfidf_consensus3", lexical),
            "refined_frozen": refined,
        }
        print(
            doc,
            len(pages),
            "pages",
            len(windows),
            "windows; predictions saved without display",
            flush=True,
        )
    write(output / "predictions.json", predictions)
    write(
        output / "prediction_lock.json",
        dict(
            saved_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_manifest_sha256=sha((ROOT / "sources.json").read_bytes()),
            inference_sha256=sha(Path(__file__).read_bytes()),
            predictions_sha256=sha((output / "predictions.json").read_bytes()),
            prior_freeze_sha256=sha((ROOT / "freeze.json").read_bytes()),
            reference_existed_at_run=(ROOT / "reference.json").exists(),
            elapsed_seconds=round(time.monotonic() - start, 3),
            new_fits=0,
            interpretation="Unchanged-reader first application, not an independent human benchmark; search snippets seen.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", default=ROOT / "results")
    args = p.parse_args()
    run(args.cache, args.model, args.output)
