"""Verify first-response provenance, frozen methods, all scores and source positions."""

import argparse
import contextlib
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from core import sha, decision
from infer import ROOT, MODELS, request
from evaluate import score
from prepare_inputs import public_stub, build, canonical
from report import export, run as report
from diagnostics import calculate, repairs
from provenance import evaluate as provenance
from contract_report import calculate as contract_score
from contract_infer import contract_request

DIGESTS = {
    "qwen": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
    "kanana-public": "b1b93e15aca0378631ccedfa74dee888ac3b7e1a551d379edfa8bc33a402a89d",
}
SPLITS = ["development", "known", "curated", "retrieved"]


def read(name):
    return json.loads((ROOT / name).read_text())


def stamp(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def public_checks():
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py")
    tests = unittest.TextTestRunner().run(suite)
    assert tests.wasSuccessful()
    freezes = {}
    for path in sorted(ROOT.glob("*freeze.json")):
        f = json.loads(path.read_text())
        for name, h in f["artifacts"].items():
            assert sha((ROOT / name).read_bytes()) == h, (path.name, name)
        freezes[path.name] = f["created_utc"]
    old = 0
    for folder in [
        "policy_reading_v3",
        "policy_reading_v4",
        "policy_reading_v5",
        "policy_semantic_diff",
        "policy_condition_reading",
        "policy_condition_transfer",
        "policy_revision_audit",
        "policy_scope_reading",
    ]:
        p = ROOT.parent / folder
        for n, h in json.loads((p / "research_record.json").read_text())[
            "artifacts"
        ].items():
            assert sha((p / n).read_bytes()) == h, (folder, n)
            old += 1
    calls = 0
    groups = 0
    latest = {}
    for split in SPLITS:
        reference = read(
            "development.json" if split == "development" else split + "_reference.json"
        )
        data = reference if split == "development" else public_stub(reference)
        expected = read("results/" + split + "/case_scores.json")
        summary = read("results/" + split + "/summary.json")
        for model in MODELS:
            records = read("results/" + split + "/" + model + ".json")
            actual = score(data, records)
            assert actual == expected[model], (split, model, "case replay")
            assert {m: v["metrics"] for m, v in actual.items()} == summary[model]
            groups += len(actual)
            freeze = freezes[
                (
                    "development_freeze.json"
                    if split == "development"
                    else "known_freeze.json" if split == "known" else "test_freeze.json"
                )
            ]
            for stage, rows in records.items():
                for r in rows.values():
                    assert r["model_digest"] == DIGESTS[model], (split, stage)
                    assert r["model_name"] == MODELS[model]
                    assert stamp(r["created_utc"]) >= stamp(freeze)
                    assert r["response_redacted"] and "response" not in r
                    assert (
                        r["runtime"].get("prompt_eval_count", 0) < 8192
                    ), "Possible context truncation"
                    calls += 1
                    latest[split] = max(
                        latest.get(split, r["created_utc"]), r["created_utc"]
                    )
            for c in data["cases"]:
                assert stamp(records["before"][c["id"]]["created_utc"]) <= stamp(
                    records["after"][c["id"]]["created_utc"]
                )
                for stage in ["direct_gate", "map_gate"]:
                    assert stamp(records[stage][c["id"]]["created_utc"]) >= stamp(
                        records["before"][c["id"]]["created_utc"]
                    )
                    assert stamp(records[stage][c["id"]]["created_utc"]) >= stamp(
                        records["map"][c["bundle"]]["created_utc"]
                    )
        assert calculate(split) == read("results/" + split + "/diagnostics.json")
        if split in ["curated", "retrieved"]:
            assert provenance(reference, expected) == read(
                "results/" + split + "/provenance.json"
            )
    repair = read("results/retrieval_repairs.json")
    for split in ["top5", "neighbor5"]:
        ref = read(split + "_reference.json")
        bs = {b["id"]: b for b in ref["bundles"]}
        for model in MODELS:
            records = read("results/" + split + "/" + model + ".json")
            assert len(records) == 32
            for r in records.values():
                assert stamp(r["created_utc"]) >= stamp(freezes["repair_freeze.json"])
                assert r["model_digest"] == DIGESTS[model]
                assert r["runtime"].get("prompt_eval_count", 0) < 8192
                calls += 1
            rows = repair[split][model]["cases"]
            for r, c in zip(rows, ref["cases"]):
                loc = bs[c["bundle"]]["source_locators"]
                pred = decision(
                    records[c["id"]]["parsed"], {**loc["before"], **loc["after"]}
                )
                assert r["prediction"] == pred and r["correct"] == (
                    pred["decision"] == c["reference"]["after"]
                )
            for key in ["correct", "joint", "cited_full_span", "supplied_full_span"]:
                assert sum(r[key] for r in rows) == repair[split][model]["metrics"][key]
    from support_sensitivity import calculate as sensitivity

    assert sensitivity() == read("results/support_sensitivity.json")
    contract = read("results/contract/predictions.json")
    assert contract_score(contract) == read("results/contract/summary.json")
    for stages in contract.values():
        for rows in stages.values():
            for r in rows.values():
                assert stamp(r["created_utc"]) >= stamp(freezes["contract_freeze.json"])
                assert r["model_digest"] == DIGESTS[r["model"]]
                calls += 1
    with (ROOT / "human_review.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) == 62
        assert all(
            not r[k]
            for r in rows
            for k in ["reviewer", "before_label", "after_label", "route", "notes"]
        )
    manifest = ROOT / "research_record.json"
    artifacts = 0
    if manifest.exists():
        for n, h in json.loads(manifest.read_text())["artifacts"].items():
            assert sha((ROOT / n).read_bytes()) == h, n
            artifacts += 1
    return {
        "tests": tests.testsRun,
        "freezes": len(freezes),
        "previous_artifacts": old,
        "score_groups": groups,
        "first_responses": calls,
        "artifacts": artifacts,
        "latest_response_utc": latest,
    }


def private_checks(cache):
    cache = Path(cache)
    inputs = {
        "development": read("development.json"),
        "known": json.loads((cache / "known_inputs.json").read_text()),
    }
    for split in ["curated", "retrieved", "top5", "neighbor5"]:
        inputs[split] = build(cache, split)
        assert inputs[split] == json.loads(
            (cache / (split + "_inputs.json")).read_text()
        )
    for split in ["curated", "retrieved"]:
        assert (
            sha(inputs[split])
            == read("test_freeze.json")["private_input_sha256"][split]
        )
    for split in ["top5", "neighbor5"]:
        assert (
            sha(inputs[split])
            == read("repair_freeze.json")["private_input_sha256"][split]
        )
    assert sha(inputs["known"]) == read("known_freeze.json")["private_input_sha256"]
    # Rerun retrieval from full pages; annotations never select chunks.
    from retrieve import chunks, select

    sources = {
        s["id"]: canonical(cache / (s["id"] + ".pdf")) for s in read("sources.json")
    }
    for b, c in zip(
        read("retrieved_reference.json")["bundles"], inputs["retrieved"]["cases"]
    ):
        for phase, prefix in [("before", "O"), ("after", "N")]:
            loc = b["source_locators"][phase]
            doc = next(iter(loc.values()))["document"]
            selected = select(c["claim"], chunks(sources[doc], prefix))
            assert [r["id"] for r in selected] == list(loc)
    from expand_retrieval import build as expand

    for split in ["top5", "neighbor5"]:
        data, ref = expand(cache, split)
        assert data == inputs[split] and ref == read(split + "_reference.json")
    for loc in read("support_sensitivity.json")["after_support"].values():
        value = sources[loc["document"]][loc["page"] - 1][loc["start"] : loc["end"]]
        assert sha(value.encode()) == loc["text_sha256"]
    requests_checked = 0
    exports = 0
    for split in SPLITS:
        data = inputs[split]
        bs = {b["id"]: b for b in data["bundles"]}
        cs = {c["id"]: c for c in data["cases"]}
        for model in MODELS:
            pub = read("results/" + split + "/" + model + ".json")
            raw = {
                s: {
                    p.stem: json.loads(p.read_text())
                    for p in (cache / split / model / s).glob("*.json")
                }
                for s in pub
            }
            for stage, rows in raw.items():
                for key, r in rows.items():
                    if stage == "map":
                        req = request(bs[key], None, model, stage)
                    else:
                        case = cs[key]
                        b = bs[case["bundle"]]
                        old = decision(raw["before"][key]["parsed"], b["before"])
                        req = request(
                            b, case, model, stage, old, raw["map"][b["id"]]["parsed"]
                        )
                    assert (
                        sha(req) == r["request_sha256"]
                        and sha(req["messages"]) == r["input_sha256"]
                    ), (split, model, stage, key)
                    assert export(r) == pub[stage][key]
                    requests_checked += 1
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                report(
                    (
                        ROOT / "development.json"
                        if split == "development"
                        else cache / (split + "_inputs.json")
                    ),
                    cache / split,
                    split,
                    temporary,
                )
            for p in Path(temporary).glob("*.json"):
                assert (
                    p.read_bytes() == (ROOT / "results" / split / p.name).read_bytes()
                )
                exports += 1
    for split in ["top5", "neighbor5"]:
        data = inputs[split]
        bs = {b["id"]: b for b in data["bundles"]}
        for model in MODELS:
            pub = read("results/" + split + "/" + model + ".json")
            for c in data["cases"]:
                r = json.loads(
                    (cache / split / model / (c["id"] + ".json")).read_text()
                )
                req = request(bs[c["bundle"]], c, model, "after")
                assert (
                    sha(req) == r["request_sha256"]
                    and sha(req["messages"]) == r["input_sha256"]
                )
                assert export(r) == pub[c["id"]]
                requests_checked += 1
    assert repairs(cache) == read("results/retrieval_repairs.json")
    b = next(b for b in inputs["curated"]["bundles"] if b["id"] == "seoul")
    cap = {
        p: {k: v for k, v in b[p].items() if k.endswith("CAP")}
        for p in ["before", "after"]
    }
    contract = read("results/contract/predictions.json")
    for model, stages in contract.items():
        for stage, rows in stages.items():
            for key, pub in rows.items():
                r = json.loads(
                    (cache / "contract" / model / stage / (key + ".json")).read_text()
                )
                if stage == "contract":
                    req = contract_request(cap[key], model)
                else:
                    case = next(c for c in inputs["curated"]["cases"] if c["id"] == key)
                    req = request(cap, case, model, "after")
                assert (
                    sha(req) == r["request_sha256"]
                    and sha(req["messages"]) == r["input_sha256"]
                )
                assert export(r) == pub
                requests_checked += 1
    return {
        "private_requests_rebuilt": requests_checked,
        "report_files_byte_replayed": exports,
        "new_sources": len(sources),
        "retrieval_replayed": 3 * 32 * 2,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    a = p.parse_args()
    result = public_checks()
    if a.cache:
        result.update(private_checks(a.cache))
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, indent=2))
