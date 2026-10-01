"""Replay public scores, source coordinates, raw provenance and immutable lineage."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Load this package's source builder before bridge adds the prior package path.
from prepare_inputs import build
from bridge import ROOT, PREVIOUS, MODELS, sha, request, interpret, adapt
from report import score, export
from rescore_previous import SPECS, calculate
from counterexamples import witness
from witness_audit import calculate as audit_witnesses

DIGESTS = {
    "qwen": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
    "kanana-public": "b1b93e15aca0378631ccedfa74dee888ac3b7e1a551d379edfa8bc33a402a89d",
}


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    p.add_argument("--out")
    a = p.parse_args()
    cache = Path(a.cache) if a.cache else None
    tests = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py")
    )
    assert tests.wasSuccessful()
    freezes = {}
    for path in ROOT.glob("*_freeze.json"):
        record = read(path)
        for name, digest in record["files"].items():
            assert sha((ROOT / name).read_bytes()) == digest, (path.name, name)
        freezes[path.stem] = datetime.fromisoformat(record["created_utc"])
    old_count = 0
    for name in [
        "policy_reading_v3",
        "policy_reading_v4",
        "policy_reading_v5",
        "policy_semantic_diff",
        "policy_condition_reading",
        "policy_condition_transfer",
        "policy_revision_audit",
        "policy_scope_reading",
        "policy_selective_update",
        "policy_compositional_reasoning",
    ]:
        package = ROOT.parent / name
        for file, digest in read(package / "research_record.json")["artifacts"].items():
            assert sha((package / file).read_bytes()) == digest, (name, file)
            old_count += 1
    new_count = raw_count = replay_count = 0
    unique = set()
    for split, ref in [
        ("metamorphic", "metamorphic.json"),
        ("official", "official_reference.json"),
    ]:
        data = read(ROOT / ref)
        if split == "official" and cache:
            data = build(cache / "sources", data)
            assert data == read(cache / "official_inputs.json")
        records = read(ROOT / f"results/{split}/predictions.json")
        report = score(data, records)
        for name, result in report.items():
            assert result == read(ROOT / f"results/{split}/{name}.json"), (split, name)
            replay_count += 1
        bs = {b["id"]: b for b in data["bundles"]}
        cs = {c["id"]: c for c in data["cases"]}
        assert len(records) == len(MODELS) * (len(bs) + 2 * len(cs))
        for r in records:
            assert r["model_name"] == MODELS[r["model"]]
            assert r["model_digest"] == DIGESTS[r["model"]]
            assert not {"response", "messages", "prompt"} & r.keys()
            t = datetime.fromisoformat(r["created_utc"])
            assert t >= freezes["method_freeze"]
            if split == "official":
                assert t >= freezes["official_freeze"]
            unique.add(
                (
                    r["model"],
                    r["request_sha256"],
                    r["created_utc"],
                    r["response_sha256"],
                )
            )
            new_count += 1
            if cache:
                raw = read(
                    cache / split / r["model"] / r["stage"] / (r["id"] + ".json")
                )
                c = None if r["stage"] == "policy" else cs[r["id"]]
                b = bs[r["id"]] if c is None else bs[c["bundle"]]
                req = request(b, c, r["model"], r["stage"])
                assert r["request_sha256"] == raw["request_sha256"] == sha(req)
                assert r["input_sha256"] == raw["input_sha256"] == sha(req["messages"])
                assert r["response_sha256"] == sha(raw["response"])
                for k in ["parsed", "model_digest", "created_utc", "elapsed_seconds"]:
                    assert r[k] == raw[k]
                assert (
                    raw["response"].get("prompt_eval_count", 0)
                    < req["options"]["num_ctx"]
                )
                raw_count += 1
        if cache:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                inp = tmp / "input.json"
                inp.write_text(json.dumps(data, ensure_ascii=False))
                with contextlib.redirect_stdout(io.StringIO()):
                    export(inp, cache / split, tmp / "output")
                for name in ["predictions", *report]:
                    assert (tmp / f"output/{name}.json").read_bytes() == (
                        ROOT / f"results/{split}/{name}.json"
                    ).read_bytes()
    assert new_count == len(unique) == 216
    for split in SPECS:
        for name, result in calculate(split).items():
            assert result == read(ROOT / f"results/previous_{split}/{name}.json")
            replay_count += 1
    witness_count = 0
    assert audit_witnesses() == read(ROOT / "results/counterexamples_audited.json")
    for r in read(ROOT / "results/counterexamples.json"):
        data = read(PREVIOUS / SPECS[r["split"]])
        b = next(b for b in data["bundles"] if b["id"] == r["bundle"])
        preds = read(PREVIOUS / f"results/{r['split']}/predictions.json")
        p = next(
            p["parsed"]
            for p in preds
            if p["stage"] == "policy"
            and p["model"] == r["model"]
            and p["id"] == r["bundle"]
        )
        assert (
            witness(p, b["reference_policy"], list(b["evidence"]))["status"]
            == r["status"]
        )
        if r["status"] == "witness":
            for name, policy in [
                ("predicted", p),
                ("reference", b["reference_policy"]),
            ]:
                assert (
                    interpret(adapt(policy), r["profile"], list(b["evidence"]))[
                        "decision"
                    ]
                    == r[name]
                )
            assert r["predicted"] != r["reference"]
            witness_count += 1
    for s in read(ROOT / "sources.json"):
        assert (
            freezes["method_freeze"]
            < datetime.fromisoformat(s["retrieved_utc"])
            < freezes["official_freeze"]
        )
    manifest = read(ROOT / "research_record.json")
    for name, digest in manifest["artifacts"].items():
        assert sha((ROOT / name).read_bytes()) == digest, name
    result = dict(
        status="PASS",
        tests=tests.testsRun,
        freezes=len(freezes),
        old_artifacts=old_count,
        new_responses=new_count,
        raw_verified=raw_count,
        replayed_reports=replay_count,
        disagreement_witnesses=witness_count,
        artifacts=len(manifest["artifacts"]),
    )
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
