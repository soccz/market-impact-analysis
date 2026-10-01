"""Replay published decisions and optionally audit complete private provenance."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import unittest

from experiment import MODELS, request, sha
from method import decide, literal_rules, retrieve
from rehydrate import reconstruct
from report import load, score, semantic_audit, missing_summary
from test_reference import grid_check

ROOT = Path(__file__).parent
DIGESTS = dict(
    qwen="6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
    **{
        "kanana-public": "b1b93e15aca0378631ccedfa74dee888ac3b7e1a551d379edfa8bc33a402a89d"
    },
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private", help="Private root containing source HTML, inputs.json and cache/"
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    suite = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py")
    )
    assert suite.wasSuccessful()
    freezes = {}
    for path in ROOT.glob("*_freeze.json"):
        data = load(path)
        freezes[path.stem] = datetime.fromisoformat(data["created_utc"])
        for name, digest in data["files"].items():
            assert sha((ROOT / name).read_bytes()) == digest, name
    assert freezes["method_freeze"] < freezes["evaluation_freeze"]
    old_artifacts = 0
    for directory in ROOT.parent.glob("policy_*"):
        if directory == ROOT or not (directory / "research_record.json").exists():
            continue
        for name, digest in load(directory / "research_record.json")[
            "artifacts"
        ].items():
            assert sha((directory / name).read_bytes()) == digest, (
                directory.name,
                name,
            )
            old_artifacts += 1
    public = load(ROOT / "reference.json")
    docs = {d["id"]: d for d in public["documents"]}
    cases = {c["id"]: c for c in public["cases"]}
    assert len(docs) == 6 and len(cases) == 48
    for d in docs.values():
        for c in cases.values():
            if c["document"] == d["id"]:
                assert (
                    decide(
                        d["reference_program"], c["facts"], d["fields"], d["evidence"]
                    )["decision"]
                    == c["label"]
                )
    predictions = load(ROOT / "results/predictions.json")
    assert len(predictions) == 120
    assert len({(p["model"], p["stage"], p["id"]) for p in predictions}) == 120
    assert len({(p["model"], p["request_sha256"]) for p in predictions}) == 120
    for p in predictions:
        assert p["model_name"] == MODELS[p["model"]]
        assert p["model_digest"] == DIGESTS[p["model"]]
        assert datetime.fromisoformat(p["created_utc"]) > freezes["evaluation_freeze"]
        assert p["response_sha256"] and p["request_sha256"] and p["input_sha256"]
        assert not {"response", "messages", "prompt"} & p.keys()
    rules = load(ROOT / "results/rules.json")
    retrieval = load(ROOT / "results/retrieval.json")
    for key, value in retrieval.items():
        required = set(docs[key]["required_evidence"])
        assert value["required"] == sorted(required)
        assert value["total_paragraphs"] == len(docs[key]["evidence"])
        for kind in ["ranked", "expanded"]:
            assert value[kind + "_recall"] == len(required & set(value[kind])) / len(
                required
            )
    costs = load(ROOT / "results/costs.json")
    for model in MODELS:
        for stage in ["direct", "full", "retrieved"]:
            selected = [
                p for p in predictions if p["model"] == model and p["stage"] == stage
            ]
            assert costs[model][stage] == dict(
                requests=len(selected),
                prompt_tokens=sum(p["prompt_tokens"] for p in selected),
                output_tokens=sum(p["output_tokens"] for p in selected),
                elapsed_seconds=round(sum(p["elapsed_seconds"] for p in selected), 3),
                unusable=sum(not p["usable"] for p in selected),
            )
    rows, summary, pairs = score(public, predictions, rules, retrieval)
    assert missing_summary(rows) == load(ROOT / "results/missing.json")
    for name, value in [("rows", rows), ("summary", summary), ("pairs", pairs)]:
        assert value == load(ROOT / f"results/{name}.json"), name
    programs = load(ROOT / "results/programs.json")
    for item in programs:
        d = docs[item["document"]]
        p = next(
            p
            for p in predictions
            if (p["id"], p["model"], p["stage"])
            == (item["document"], item["model"], item["stage"])
        )
        evidence = (
            d["evidence"]
            if item["stage"] == "full"
            else {k: d["evidence"][k] for k in retrieval[d["id"]]["expanded"]}
        )
        value = semantic_audit(
            p["parsed"] if p["usable"] else None,
            d["reference_program"],
            d["fields"],
            evidence,
        )
        assert value == {
            k: v for k, v in item.items() if k not in ["document", "model", "stage"]
        }
    for source in load(ROOT / "sources.json"):
        assert source["tls_verified"]
        assert (
            freezes["method_freeze"]
            < datetime.fromisoformat(source["retrieved_utc"])
            < freezes["evaluation_freeze"]
        )
    reviews = list(csv.DictReader((ROOT / "human_review.csv").open()))
    assert len(reviews) == 48 and {r["id"] for r in reviews} == set(cases)
    for review in reviews:
        assert not any(review[k] for k in review if k not in ["id", "document"])
    raw_count = 0
    if args.private:
        directory = Path(args.private)
        actual = reconstruct(directory)
        assert actual == load(directory / "inputs.json")
        assert (
            sha((directory / "inputs.json").read_bytes())
            == load(ROOT / "evaluation_freeze.json")["private_input_sha256"]
        )
        actual_docs = {d["id"]: d for d in actual["documents"]}
        for key, d in actual_docs.items():
            rule, guarded = literal_rules(d)
            assert rules[key] == dict(program=rule, guarded=guarded)
            calculated = retrieve(d)
            assert all(retrieval[key][k] == v for k, v in calculated.items())
        for p in predictions:
            record = load(
                directory / "cache" / p["model"] / p["stage"] / (p["id"] + ".json")
            )
            raw = record["response"]
            case = cases[p["id"]] if p["stage"] == "direct" else None
            d = actual_docs[case["document"] if case else p["id"]]
            req = request(d, case, p["model"], p["stage"])
            assert record["request_sha256"] == p["request_sha256"] == sha(req)
            assert record["input_sha256"] == p["input_sha256"] == sha(req["messages"])
            assert record["response_sha256"] == p["response_sha256"] == sha(raw)
            assert record["model_digest"] == p["model_digest"]
            assert record["created_utc"] == p["created_utc"]
            assert record["parsed"] == p["parsed"]
            try:
                parsed = json.loads(raw["message"]["content"])
            except json.JSONDecodeError:
                parsed = None
            assert parsed == p["parsed"]
            assert p["usable"] == (
                raw.get("done") is True
                and raw.get("done_reason") == "stop"
                and raw.get("prompt_eval_count", 0) + raw.get("eval_count", 0) < 16384
            )
            raw_count += 1
    manifest = load(ROOT / "research_record.json")
    for name, digest in manifest["artifacts"].items():
        assert sha((ROOT / name).read_bytes()) == digest, name
    report = dict(
        status="PASS",
        tests=suite.testsRun,
        independent_predicate_grid=grid_check(),
        freezes=len(freezes),
        old_artifacts=old_artifacts,
        public_predictions=120,
        private_raw_records=raw_count,
        case_rows=len(rows),
        graph_audits=len(programs),
        blank_human_reviews=48,
        artifact_count=len(manifest["artifacts"]),
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
