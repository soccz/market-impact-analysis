"""Replay deterministic results and verify source/evidence integrity."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluate import evaluate
from prepare import load
from reader import FIELDS, chunks, predict, read_source, sha, tfidf_ranks
from posthoc import run as run_posthoc
from refined import run as run_refined
from bridge import run as run_bridge

ROOT = Path(__file__).resolve().parent


def verify(cache):
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_reader.py"], cwd=ROOT, check=True
    )
    frozen = json.loads((ROOT / "freeze.json").read_text())
    for name, expected in frozen["files"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    record = ROOT / "research_record.json"
    if record.exists():
        for name, expected in json.loads(record.read_text())["artifacts"].items():
            assert sha((ROOT / name).read_bytes()) == expected, name
    docs = load(cache)
    refs = json.loads((ROOT / "reference.json").read_text())
    for row in refs:
        for group in row["support_groups"]:
            for span in group:
                text = docs[row["document"]][span["page"] - 1][
                    span["start"] : span["end"]
                ]
                assert sha(text.encode()) == span["sha256"]
    predictions = json.loads((ROOT / "results/predictions.json").read_text())
    spans_checked = 0
    for doc, methods in predictions.items():
        pages = docs[doc]
        ranking = tfidf_ranks(chunks(pages))
        for method, fields in methods.items():
            source_pages = (
                read_source(Path(cache) / (doc + ".bin"), layout=False)
                if method == "unsorted_tfidf_top1"
                else pages
            )
            for field, result in fields.items():
                for span in result["evidence"]:
                    text = source_pages[span["page"] - 1][span["start"] : span["end"]]
                    assert sha(text.encode()) == span["sha256"], (doc, method, field)
                    spans_checked += 1
            if method != "klue_top1":
                rank = (
                    tfidf_ranks(chunks(source_pages))
                    if method == "unsorted_tfidf_top1"
                    else ranking
                )
                actual_method = (
                    "tfidf_top1" if method == "unsorted_tfidf_top1" else method
                )
                assert predict(source_pages, actual_method, rank) == fields, (
                    doc,
                    method,
                )
    # A stricter, independent containment audit gives the same joint counts as
    # the frozen evaluator (whose one-character endpoint tolerance is unused).
    summary = json.loads((ROOT / "results/summary.json").read_text())
    for method, counts in summary["metrics"].items():
        strict_joint = 0
        for r in refs:
            p = predictions[r["document"]][method][r["field"]]
            grounded = bool(r["support_groups"]) and all(
                any(
                    any(
                        s["page"] == e["page"]
                        and e["start"] <= s["start"]
                        and e["end"] >= s["end"]
                        for e in p["evidence"]
                    )
                    for s in group
                )
                for group in r["support_groups"]
            )
            strict_joint += (
                r["value"] is not None and p["value"] == r["value"] and grounded
            )
        assert strict_joint == counts["correct_joint"]
    with tempfile.TemporaryDirectory(prefix="condition-reading-verify-") as temp:
        temp = Path(temp)
        evaluate(ROOT / "results/predictions.json", temp)
        for name in [
            "summary.json",
            "errors.json",
            "pairs.json",
            "profiles.json",
            "gates.json",
            "RESULTS.md",
        ]:
            assert (temp / name).read_bytes() == (
                ROOT / "results" / name
            ).read_bytes(), name
        for name, runner in [
            ("posthoc.json", run_posthoc),
            ("refined.json", run_refined),
        ]:
            runner(cache, temp / name)
            assert (temp / name).read_bytes() == (
                ROOT / "results" / name
            ).read_bytes(), name
            for doc, result in json.loads((temp / name).read_text())[
                "predictions"
            ].items():
                for value in result["fields"].values():
                    for span in value["evidence"]:
                        text = docs[doc][span["page"] - 1][span["start"] : span["end"]]
                        assert sha(text.encode()) == span["sha256"]
        run_bridge(temp / "bridge.json")
        assert (temp / "bridge.json").read_bytes() == (
            ROOT / "results/bridge.json"
        ).read_bytes()
    # Preserve every previously published research artifact with a manifest.
    preserved = 0
    for directory in [
        "policy_reading_v3",
        "policy_reading_v4",
        "policy_reading_v5",
        "policy_semantic_diff",
    ]:
        path = ROOT.parent / directory / "research_record.json"
        if path.exists():
            prior = json.loads(path.read_text())
            for name, expected in prior.get("artifacts", {}).items():
                if isinstance(expected, str):
                    assert sha((path.parent / name).read_bytes()) == expected, name
                    preserved += 1
    print(
        json.dumps(
            dict(
                success=True,
                official_sources=8,
                provisional_fields=32,
                source_evidence_spans=spans_checked,
                result_files_replayed=9,
                previous_artifacts_preserved=preserved,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    verify(args.cache)
