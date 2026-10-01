"""Public score replay; optional private source/request/response audit."""

import argparse
import copy
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from bridge import ROOT, FACT, MODELS, sha, profile_exact, adapt
from candidates import compile_selection
from explicit_logic import as_selection
from evaluate_v2 import (
    execute,
    development_data,
    previous_rows,
    previous_records,
    evaluate,
)
from extended_logic import equivalent
from claim_alignment import align
from report_transfer import summary as transfer_summary, public_record
from revision_chain import UPDATE, validate, answer, update, claim_slots
import ast
from fractions import Fraction

# Reuse the frozen independent arithmetic function without importing its old
# package-level bridge (whose module name is shared across historical studies).
_tree = ast.parse((FACT / "witness_audit.py").read_text())
_function = next(
    x
    for x in _tree.body
    if isinstance(x, ast.FunctionDef) and x.name == "complete_decision"
)
_namespace = {"Fraction": Fraction}
exec(
    compile(
        ast.Module(body=[_function], type_ignores=[]), "frozen_fraction_audit", "exec"
    ),
    _namespace,
)
complete_decision = _namespace["complete_decision"]

DIGESTS = {
    "qwen": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
    "kanana-public": "b1b93e15aca0378631ccedfa74dee888ac3b7e1a551d379edfa8bc33a402a89d",
}
DEVELOPMENT = ["development", "clarification", "repair", "explicit_logic"]


def load(path):
    return json.loads(Path(path).read_text())


def dev_summary(rows):
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        out[ds] = {}
        for m in MODELS:
            rr = [r for r in rows if r["model"] == m and r["dataset"] == ds]
            out[ds][m] = {}
            for method in ["direct", "previous", "candidate_rules", "candidate_model"]:
                out[ds][m][method] = dict(
                    n=len(rr),
                    correct=sum(r[method] == r["label"] for r in rr),
                    joint=sum(
                        r[method] == r["label"] and r["profile_exact"] for r in rr
                    ),
                    abstain=sum(r[method] == "abstain" for r in rr),
                    fixes=sum(
                        r[method] == r["label"] and r["previous"] != r["label"]
                        for r in rr
                    ),
                    harms=sum(
                        r[method] != r["label"] and r["previous"] == r["label"]
                        for r in rr
                    ),
                )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    p.add_argument("--out")
    a = p.parse_args()
    cache = Path(a.cache) if a.cache else None
    suite = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py")
    )
    assert suite.wasSuccessful()
    freezes = {}
    for path in ROOT.glob("*_freeze.json"):
        data = load(path)
        freezes[path.stem] = datetime.fromisoformat(data["created_utc"])
        for f, digest in data["files"].items():
            assert sha((ROOT / f).read_bytes()) == digest, (path.name, f)
    old_count = 0
    for directory in ROOT.parent.glob("policy_*"):
        if directory == ROOT or not (directory / "research_record.json").exists():
            continue
        for f, digest in load(directory / "research_record.json")["artifacts"].items():
            assert sha((directory / f).read_bytes()) == digest, (directory.name, f)
            old_count += 1
    assert old_count == 2354, old_count
    dev = dict(bundles=[], cases=[])
    for file in [FACT / "metamorphic.json", FACT / "official_reference.json"]:
        d = load(file)
        dev["bundles"] += d["bundles"]
        dev["cases"] += d["cases"]
    dc = {c["id"]: c for c in dev["cases"]}
    db = {b["id"]: b for b in dev["bundles"]}
    old = {(r["model"], r["id"]): r for r in previous_rows()}
    public_unique = {}
    reports = 0
    for path in (ROOT / "results").glob("*/predictions.json"):
        for r in load(path):
            assert (
                r["model_name"] == MODELS[r["model"]]
                and r["model_digest"] == DIGESTS[r["model"]]
            )
            assert not {"messages", "response", "prompt"} & r.keys()
            key = (
                r["model"],
                r["request_sha256"],
                r["created_utc"],
                r["response_sha256"],
            )
            public_unique[key] = r
    for split in DEVELOPMENT:
        base = ROOT / "results" / split
        rows = load(base / "rows.json")
        rules = load(base / "rules.json")
        pred = {
            (r["model"], r["id"]): r["parsed"] for r in load(base / "predictions.json")
        }
        for rule in rules:
            b = db[rule["bundle"]]
            value = pred[rule["model"], b["id"]]
            value = as_selection(value) if split == "explicit_logic" else value
            assert compile_selection(value, rule["catalog"]) == rule["model_policy"]
            for method in ["model", "rule"]:
                assert (
                    equivalent(
                        rule[method + "_policy"],
                        adapt(b["reference_policy"]),
                        list(b["evidence"]),
                    )
                    == rule[method + "_equivalent"]
                )
        for row in rows:
            c = dc[row["id"]]
            b = db[c["bundle"]]
            r = next(
                r
                for r in rules
                if r["model"] == row["model"] and r["bundle"] == b["id"]
            )
            prior = old[row["model"], row["id"]]
            assert (
                row["profile"] == prior["corrected_profile"]
                and row["direct"] == prior["direct"]
                and row["previous"] == prior["corrected"]
            )
            assert row["profile_exact"] == profile_exact(
                row["profile"], c["reference_profile"]
            )
            for method in ["rules", "model"]:
                assert row["candidate_" + method] == execute(
                    r["rule_policy" if method == "rules" else "model_policy"],
                    row["profile"],
                    list(b["evidence"]),
                )
        assert dev_summary(rows) == load(base / "summary.json")
        reports += 3
    aligned = load(ROOT / "results/aligned/rows.json")
    summary = {}
    for row in aligned:
        c = dc[row["id"]]
        facts, audit = align(c["claim"], row["profile"])
        assert facts == row["aligned_profile"] and audit == row["alignment"]
        assert (
            profile_exact(facts, c["reference_profile"]) == row["aligned_profile_exact"]
        )
        r = next(
            r
            for r in load(ROOT / "results/explicit_logic/rules.json")
            if r["model"] == row["model"] and r["bundle"] == row["bundle"]
        )
        assert (
            execute(r["model_policy"], facts, list(db[row["bundle"]]["evidence"]))
            == row["aligned"]
        )
    for ds in ["metamorphic", "official"]:
        summary[ds] = {}
        for m in MODELS:
            rr = [r for r in aligned if r["model"] == m and r["dataset"] == ds]
            summary[ds][m] = dict(
                n=len(rr),
                correct=sum(r["aligned"] == r["label"] for r in rr),
                profile_exact=sum(r["aligned_profile_exact"] for r in rr),
                joint=sum(
                    r["aligned"] == r["label"] and r["aligned_profile_exact"]
                    for r in rr
                ),
                abstain=sum(r["aligned"] == "abstain" for r in rr),
            )
    assert summary == load(ROOT / "results/aligned/summary.json")
    reports += 2
    for split in ["transfer", "transfer_repair", "confirmation"]:
        base = ROOT / "results" / split
        ref = load(
            ROOT
            / (
                "confirmation_reference.json"
                if split == "confirmation"
                else "transfer_reference.json"
            )
        )
        cs = {c["id"]: c for c in ref["cases"]}
        rows = load(base / "rows.json")
        rules = load(base / "rules.json")
        for r in rules:
            for method in ["rule", "model"] + (
                ["freeform"] if "freeform_policy" in r else []
            ):
                assert (
                    equivalent(
                        r[method + "_policy"], r["reference_policy"], r["allowed"]
                    )
                    == r[method + "_equivalent"]
                )
        for row in rows:
            r = next(
                r
                for r in rules
                if r["model_id"] == row["model_id"] and r["bundle"] == row["bundle"]
            )
            c = cs[row["id"]]
            facts, audit = align(c["claim"], row["raw_profile"])
            assert facts == row["profile"] and audit == row["alignment"]
            assert row["profile_exact"] == profile_exact(
                facts, c["reference_profile"]
            ) and row["raw_profile_exact"] == profile_exact(
                row["raw_profile"], c["reference_profile"]
            )
            for method, key in [("rules", "rule_policy"), ("model", "model_policy")] + (
                [("freeform", "freeform_policy")] if "freeform_policy" in r else []
            ):
                assert execute(r[key], facts, r["allowed"]) == row[method]
                if method != "freeform":
                    assert (
                        execute(r[key], row["raw_profile"], r["allowed"])
                        == row[method + "_raw_facts"]
                    )
        assert transfer_summary(rows) == load(base / "summary.json")
        reports += 2
    feedbacks = load(ROOT / "development_repair_selection.json")
    rr = load(ROOT / "results/clarification/rules.json")
    witnesses = 0
    for f in feedbacks:
        if not f["selected"]:
            continue
        r = next(
            r for r in rr if r["model"] == f["model"] and r["bundle"] == f["bundle"]
        )
        v = f["feedback"]
        q = v["hypothetical_profile"]
        assert complete_decision(r["model_policy"], q) == v["model_program_decision"]
        assert complete_decision(r["rule_policy"], q) == v["restricted_parser_decision"]
        assert v["model_program_decision"] != v["restricted_parser_decision"]
        witnesses += 1
    assert witnesses == 7
    revision = load(ROOT / "results/revision/rows.json")
    audits = load(ROOT / "results/revision/contract_audit.json")
    for r in revision:
        contracts = {
            p: validate(
                next(
                    x["source_contract"]
                    for x in audits
                    if x["model"] == r["model"] and x["phase"] == p
                ),
                ["OCAP" if p == "before" else "NCAP"],
            )
            for p in ["before", "after"]
        }
        q = claim_slots(r["claim"])
        assert (
            q == r["grounded_slots"]
            and (q == r["reference_slots"]) == r["grounded_exact"]
        )
        expected = update(contracts["before"], contracts["after"], q, r["stored"])
        for k, v in expected.items():
            assert r[k] == v, (r["id"], k)
    reports += 1
    raw_count = 0
    if cache:
        from infer_candidates import request as initial_request
        from infer_candidates_v2 import request as clarified_request
        from repair_alignment import request as repair_request
        from explicit_logic import request as explicit_request
        from source_method import request as transfer_request
        from age_extension import request as repaired_request
        from bridge import old_request
        from revision_run import jobs, evidence
        from report_revision import calculate

        development = development_data(cache.parent / "policy_fact_grounding")
        bs = {b["id"]: b for b in development["bundles"]}
        reqs = {}
        for split, fn in [
            ("development", initial_request),
            ("clarification", clarified_request),
            ("repair", repair_request),
            ("explicit_logic", explicit_request),
        ]:
            for m in MODELS:
                for b in development["bundles"]:
                    fb = (
                        next(
                            x["feedback"]
                            for x in feedbacks
                            if x["model"] == m and x["bundle"] == b["id"]
                        )
                        if split == "repair"
                        else None
                    )
                    req = fn(b, None, m, "policy", fb)
                    reqs[split, m, "policy", b["id"]] = req
            records = load(ROOT / f"results/{split}/predictions.json")
            normalized = copy.deepcopy(records)
            if split == "explicit_logic":
                for r in normalized:
                    r["parsed"] = as_selection(r["parsed"])
            result = evaluate(
                development, normalized, previous_records(), previous_rows()
            )
            for k, v in result.items():
                assert v == load(ROOT / f"results/{split}/{k}.json"), (split, k)
        for name in ["transfer", "confirmation"]:
            data = load(cache / (name + "_inputs.json"))
            reference = load(ROOT / (name + "_reference.json"))
            ref = copy.deepcopy(reference)
            for b in ref["bundles"]:
                text = (cache / "sources" / (b["source"] + ".txt")).read_text()
                b["evidence"] = {
                    k: text[v["start"] : v["end"]] for k, v in b["source_spans"].items()
                }
                for k, v in b["source_spans"].items():
                    assert sha(b["evidence"][k].encode()) == v["sha256"]
            assert ref == data
            bs = {b["id"]: b for b in data["bundles"]}
            for split, fn in (
                [
                    (name, transfer_request),
                    ("transfer_repair", repaired_request),
                    ("transfer_freeform", old_request),
                ]
                if name == "transfer"
                else [(name, repaired_request)]
            ):
                for m in MODELS:
                    for b in data["bundles"]:
                        reqs[split, m, "policy", b["id"]] = fn(b, None, m, "policy")
                    if split in ["transfer", "confirmation"]:
                        for c in data["cases"]:
                            for stage in ["profile", "direct"]:
                                reqs[split, m, stage, c["id"]] = fn(
                                    bs[c["bundle"]], c, m, stage
                                )
        source = evidence(cache.parent / "policy_selective_update")
        for m in MODELS:
            for stage, key, req in jobs(source, m):
                reqs["revision", m, stage, key] = req
        rv = calculate(
            load(ROOT / "results/revision/predictions.json"),
            source,
            load(UPDATE / "results/contract/predictions.json"),
        )
        for k, v in rv.items():
            assert v == load(ROOT / f"results/revision/{k}.json")
        seen = set()
        for (split, m, stage, key), req in reqs.items():
            raw = load(cache / split / m / stage / (key + ".json"))
            r = public_record(raw)
            assert raw["request_sha256"] == sha(req) and raw["input_sha256"] == sha(
                req["messages"]
            ), (split, m, stage, key)
            identity = (m, r["request_sha256"], r["created_utc"], r["response_sha256"])
            assert public_unique[identity] == r
            seen.add(identity)
            assert (
                raw["response"].get("done") is True
                and raw["response"].get("done_reason") == "stop"
            )
            assert (
                raw["response"].get("prompt_eval_count", 0) < req["options"]["num_ctx"]
            )
            raw_count += 1
        assert seen == set(public_unique)
        for file in ["sources.json", "confirmation_sources.json"]:
            for s in load(ROOT / file):
                assert (
                    sha((cache / "sources" / (s["id"] + ".html")).read_bytes())
                    == s["sha256"]
                )
                assert (
                    sha((cache / "sources" / (s["id"] + ".txt")).read_bytes())
                    == s["text_sha256"]
                )
    for file, freeze in [
        ("sources.json", "method_freeze"),
        ("confirmation_sources.json", "age_extension_freeze"),
    ]:
        for s in load(ROOT / file):
            assert freezes[freeze] < datetime.fromisoformat(s["retrieved_utc"])
    assert len(public_unique) == 335, len(public_unique)
    if (ROOT / "research_record.json").exists():
        for f, digest in load(ROOT / "research_record.json")["artifacts"].items():
            assert sha((ROOT / f).read_bytes()) == digest, f
    result = dict(
        status="PASS",
        tests=suite.testsRun,
        interval_grid=448,
        freezes=len(freezes),
        old_artifacts=old_count,
        unique_new_responses=len(public_unique),
        raw_records_checked=raw_count,
        replayed_reports=reports,
        independent_feedback_witnesses=witnesses,
    )
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
