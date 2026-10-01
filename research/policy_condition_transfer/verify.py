"""Check immutable readers, timestamped locks, evidence and deterministic replay."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from evaluate import run as evaluate
from infer import load, verify_freeze
from reader import chunks, predict, sha, tfidf_ranks
from refined import read as refined_read

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def verify(cache):
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_evaluation.py"],
        cwd=ROOT,
        check=True,
    )
    verify_freeze()
    for name, expected in read("source_lock.json")["files"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    lock, annotation, erratum = [
        read(f)
        for f in [
            "results/prediction_lock.json",
            "annotation_record.json",
            "annotation_erratum.json",
        ]
    ]
    assert not lock["reference_existed_at_run"]
    assert not annotation["predictions_inspected_before_reference"]
    assert (
        sha((ROOT / "results/predictions.json").read_bytes())
        == lock["predictions_sha256"]
        == annotation["prediction_sha256"]
    )
    assert sha((ROOT / "sources.json").read_bytes()) == lock["source_manifest_sha256"]
    assert sha((ROOT / "infer.py").read_bytes()) == lock["inference_sha256"]
    assert sha((ROOT / "freeze.json").read_bytes()) == lock["prior_freeze_sha256"]
    assert (
        sha((ROOT / "audit/initial_annotation/reference.json").read_bytes())
        == annotation["reference_sha256"]
    )
    assert sha((ROOT / "reference.json").read_bytes()) == erratum["reference_sha256"]
    assert (
        datetime.fromisoformat(lock["saved_at_utc"])
        < datetime.fromisoformat(annotation["saved_at_utc"])
        < datetime.fromisoformat(erratum["saved_at_utc"])
    )
    docs, pred, refs = (
        load(cache),
        read("results/predictions.json"),
        read("reference.json"),
    )
    assert len(docs) == 4 and len(refs) == 16
    spans = 0
    for ref in refs:
        for group in ref["support_groups"]:
            for s in group:
                text = docs[ref["document"]][s["page"] - 1][s["start"] : s["end"]]
                assert text and sha(text.encode()) == s["sha256"]
                spans += 1
    for doc, methods in pred.items():
        ranks = tfidf_ranks(chunks(docs[doc]))
        for method, fields in methods.items():
            for item in fields.values():
                for s in item["evidence"]:
                    text = docs[doc][s["page"] - 1][s["start"] : s["end"]]
                    assert text and sha(text.encode()) == s["sha256"]
                    spans += 1
            if method == "refined_frozen":
                assert refined_read(docs[doc])[0] == fields
            elif method != "klue_top1":
                assert predict(docs[doc], method, ranks) == fields
    with tempfile.TemporaryDirectory() as output:
        evaluate(output)
        for name in ["summary.json", "field_scores.json"]:
            assert (Path(output) / name).read_bytes() == (
                ROOT / "results" / name
            ).read_bytes()
    with (ROOT / "human_review_blank.csv").open() as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 16 and all(
            not r["reviewer_id"] and not r["value"] for r in rows
        )
    preserved = 0
    for directory in [
        "policy_reading_v3",
        "policy_reading_v4",
        "policy_reading_v5",
        "policy_semantic_diff",
        "policy_condition_reading",
    ]:
        path = ROOT.parent / directory / "research_record.json"
        for name, expected in json.loads(path.read_text())["artifacts"].items():
            if isinstance(expected, str):
                assert sha((path.parent / name).read_bytes()) == expected, (
                    directory,
                    name,
                )
                preserved += 1
    record = ROOT / "research_record.json"
    if record.exists():
        for name, expected in read("research_record.json")["artifacts"].items():
            assert sha((ROOT / name).read_bytes()) == expected, name
    print(
        json.dumps(
            dict(
                success=True,
                documents=4,
                provisional_fields=16,
                evidence_spans=spans,
                deterministic_methods_replayed=4,
                result_files_replayed=2,
                old_artifacts_preserved=preserved,
                independent_human_reviews=0,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    verify(parser.parse_args().cache)
