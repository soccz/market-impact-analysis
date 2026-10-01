"""Verify public replay, protocol order, immutable lineage and optional raw provenance."""

import argparse, contextlib, io, json, tempfile, unittest
from datetime import datetime
from pathlib import Path
import evaluate, evaluate_capability
from infer import MODELS, sha, request as baseline_request
from infer_explicit import request as explicit_request
from infer_logic_examples import request as examples_request
from infer_extended import request as extended_request
from logic import interpret
from evaluate import equivalent, profile_equal, valid_direct, summarize
from extended_logic import interpret as ext_interpret, equivalent as ext_equivalent
from evaluate_capability import adapt, profile_exact, summarize as cap_summary
from prepare_inputs import build
from diagnostics import calculate, dataset
from fact_audit import calculate as fact_audit
from repair import builder, selection, SPECS as REPAIRS

ROOT = Path(__file__).parent
DIGESTS = {
    "qwen": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
    "kanana-public": "b1b93e15aca0378631ccedfa74dee888ac3b7e1a551d379edfa8bc33a402a89d",
}
ORDER = [
    "development",
    "development_explicit",
    "development_examples",
    "official",
    "official_examples",
    "composition",
    "composition_examples",
    "capability_base",
    "capability_extended",
    "official_repair",
    "composition_repair",
    "capability_repair",
]
FREEZES = {
    "development": "development",
    "development_explicit": "method",
    "development_examples": "logic_examples",
    "official": "official",
    "official_examples": "logic_examples",
    "composition": "composition",
    "composition_examples": "composition",
    "capability_base": "capability",
    "capability_extended": "capability",
    "official_repair": "repair",
    "composition_repair": "repair",
    "capability_repair": "repair",
}


def read(n):
    return json.loads((ROOT / n).read_text())


def request_fn(split):
    if split in REPAIRS:
        old, _, ext = REPAIRS[split]
        return builder(read(f"results/{old}/predictions.json"), ext)
    if split == "development":
        return baseline_request
    if split.endswith("examples") or split == "capability_base":
        return examples_request
    if split == "capability_extended":
        return extended_request
    return explicit_request


def replay(split):
    data = dataset(split)
    bs = {b["id"]: b for b in data["bundles"]}
    records = read(f"results/{split}/predictions.json")
    lookup = {(r["model"], r["stage"], r["id"]): r for r in records}
    rows = []
    for m in MODELS:
        for c in data["cases"]:
            b = bs[c["bundle"]]
            p = lookup[m, "policy", b["id"]]["parsed"]
            q = lookup[m, "profile", c["id"]]["parsed"]
            allowed = list(b["evidence"])
            rp = b["reference_policy"]
            rq = c["reference_profile"]
            ext = split.startswith("capability")
            if ext:
                p = adapt(p)
                fn = ext_interpret
            else:
                fn = interpret
            row = dict(
                id=c["id"],
                bundle=b["id"],
                model=m,
                label=c["label"],
                direct={
                    "decision": valid_direct(
                        lookup[m, "direct", c["id"]]["parsed"], allowed
                    )
                },
                pipeline=fn(p, q, allowed),
                reference_policy=fn(rp, q, allowed),
                reference_profile=fn(p, rq, allowed),
                both_reference=fn(rp, rq, allowed),
            )
            if ext:
                row.update(
                    profile_exact=profile_exact(q, rq),
                    policy_equivalent=ext_equivalent(p, rp, allowed),
                )
            else:
                row.update(
                    kleene=interpret(p, q, allowed, "kleene"),
                    profile_exact=profile_equal(q, rq),
                    policy_equivalence=equivalent(p, rp, allowed),
                )
            assert row["both_reference"]["decision"] == c["label"]
            rows.append(row)
    assert rows == read(f"results/{split}/rows.json"), (split, "rows")
    assert (
        cap_summary(rows) if split.startswith("capability") else summarize(rows)
    ) == read(f"results/{split}/summary.json"), (split, "summary")
    return data, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache")
    parser.add_argument("--out")
    a = parser.parse_args()
    cache = Path(a.cache) if a.cache else None
    tests = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py")
    )
    assert tests.wasSuccessful()
    frozen = {}
    for p in sorted(ROOT.glob("*_freeze.json")):
        d = json.loads(p.read_text())
        for name, h in d["files"].items():
            assert sha((ROOT / name).read_bytes()) == h, (p.name, name)
        frozen[p.stem.removesuffix("_freeze")] = datetime.fromisoformat(
            d["created_utc"]
        )
    old_count = 0
    previous_packages = [
        "policy_reading_v3",
        "policy_reading_v4",
        "policy_reading_v5",
        "policy_semantic_diff",
        "policy_condition_reading",
        "policy_condition_transfer",
        "policy_revision_audit",
        "policy_scope_reading",
        "policy_selective_update",
    ]
    for package in previous_packages:
        manifest = ROOT.parent / package / "research_record.json"
        for name, h in json.loads(manifest.read_text())["artifacts"].items():
            assert sha((manifest.parent / name).read_bytes()) == h, (
                manifest.parent.name,
                name,
            )
            old_count += 1
    assert selection() == read("repair_selection.json")
    assert fact_audit() == read("results/fact_audit.json")
    unique = set()
    entries = 0
    raw_verified = 0
    replayed_files = 0
    max_prompt = {m: 0 for m in MODELS}
    latest = {}
    for split in ORDER:
        data, records = replay(split)
        if cache and (split.startswith("official") or split.startswith("capability")):
            data = build(cache / "sources", data)
            filename = (
                "official_inputs.json"
                if split.startswith("official")
                else "capability_inputs.json"
            )
            assert data == json.loads((cache / filename).read_text()), (
                split,
                "source inputs",
            )
        bs = {b["id"]: b for b in data["bundles"]}
        cs = {c["id"]: c for c in data["cases"]}
        reqfn = request_fn(split)
        assert len(records) == 2 * (len(bs) + 2 * len(cs))
        for r in records:
            entries += 1
            assert (
                r["model_name"] == MODELS[r["model"]]
                and r["model_digest"] == DIGESTS[r["model"]]
            )
            assert "response" not in r and "messages" not in r
            k = (
                r["model"],
                r["stage"],
                r["id"],
                r["request_sha256"],
                r["created_utc"],
                r["response_sha256"],
            )
            if k not in unique:
                assert (
                    datetime.fromisoformat(r["created_utc"]) >= frozen[FREEZES[split]]
                ), (split, r["stage"], r["id"], "time")
                unique.add(k)
            latest[split] = max(latest.get(split, r["created_utc"]), r["created_utc"])
            if cache:
                raw = json.loads(
                    (
                        cache / split / r["model"] / r["stage"] / (r["id"] + ".json")
                    ).read_text()
                )
                c = None if r["stage"] == "policy" else cs[r["id"]]
                b = bs[r["id"]] if c is None else bs[c["bundle"]]
                req = reqfn(b, c, r["model"], r["stage"])
                assert r["request_sha256"] == raw["request_sha256"] == sha(req), (
                    split,
                    r["id"],
                    "request",
                )
                assert r["input_sha256"] == raw["input_sha256"] == sha(req["messages"])
                assert r["parsed"] == raw["parsed"] and r["response_sha256"] == sha(
                    raw["response"]
                )
                for name in ["created_utc", "model_digest", "elapsed_seconds"]:
                    assert r[name] == raw[name]
                count = raw["response"].get("prompt_eval_count", 0)
                max_prompt[r["model"]] = max(max_prompt[r["model"]], count)
                assert count < req["options"]["num_ctx"], (split, "context")
                raw_verified += 1
        if cache:
            with tempfile.TemporaryDirectory(prefix="composition-replay-") as temp:
                path = Path(temp)
                inp = path / "input.json"
                inp.write_text(json.dumps(data, ensure_ascii=False))
                dest = path / "output"
                if split.startswith("capability"):
                    saved = evaluate_capability.extended_request
                    try:
                        evaluate_capability.extended_request = reqfn
                        with contextlib.redirect_stdout(io.StringIO()):
                            evaluate_capability.run(
                                inp, cache / split, dest, "extended"
                            )
                    finally:
                        evaluate_capability.extended_request = saved
                else:
                    saved = evaluate.request
                    try:
                        evaluate.request = reqfn
                        with contextlib.redirect_stdout(io.StringIO()):
                            evaluate.run(inp, cache / split, dest)
                    finally:
                        evaluate.request = saved
                for name in ["rows.json", "predictions.json", "summary.json"]:
                    assert (dest / name).read_bytes() == (
                        ROOT / "results" / split / name
                    ).read_bytes(), (split, name)
                    replayed_files += 1
    d, e, account = calculate()
    assert (
        d == read("results/diagnostics.json")
        and e == read("results/explanations.json")
        and account == read("results/accounting.json")
    )
    assert len(unique) == account["unique_saved_responses"] == 626
    assert entries == account["exported_entries_including_reuse"]
    current = 0
    if (ROOT / "research_record.json").exists():
        for n, h in read("research_record.json")["artifacts"].items():
            assert sha((ROOT / n).read_bytes()) == h, n
            current += 1
    report = dict(
        success=True,
        tests=tests.testsRun,
        boolean_completion_checks=6237,
        money_grid_checks=36,
        freezes=len(frozen),
        previous_artifacts_unchanged=old_count,
        previous_packages=previous_packages,
        score_conditions=len(ORDER) * 2,
        unique_saved_responses=len(unique),
        entries_including_reuse=entries,
        raw_requests_verified=raw_verified,
        report_files_byte_replayed=replayed_files,
        max_reported_prompt_tokens=max_prompt,
        source_inputs_verified=bool(cache),
        current_artifacts=current,
    )
    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
