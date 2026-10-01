"""Evaluate expanded scopes against source-level labels, including DSL failures."""

import argparse, json
from pathlib import Path
from infer import MODELS, sha
from infer_logic_examples import request as base_request
from infer_extended import request as extended_request
from extended_logic import interpret, equivalent, parse_profile
from evaluate import valid_direct

ROOT = Path(__file__).parent
METHODS = [
    "direct",
    "pipeline",
    "reference_policy",
    "reference_profile",
    "both_reference",
]


def adapt(p):
    return dict(p, derived=p.get("derived", [])) if isinstance(p, dict) else p


def profile_exact(p, r):
    pv, pp, ps = parse_profile(p)
    rv, rp, rs = parse_profile(r)
    if ps != rs:
        return False
    if ps == "query_unsupported":
        return True
    if ps != "valid":
        return False
    return pp == rp and {k: v for k, v in pv.items() if v is not None} == {
        k: v for k, v in rv.items() if v is not None
    }


def summarize(rows):
    out = {}
    for m in MODELS:
        rr = [r for r in rows if r["model"] == m]
        if not rr:
            continue
        out[m] = {
            method: dict(
                n=len(rr),
                correct=sum(r[method]["decision"] == r["label"] for r in rr),
                abstain=sum(r[method]["decision"] == "abstain" for r in rr),
                unknown=sum(r[method]["decision"] == "not_established" for r in rr),
                false_determination=sum(
                    r["label"] == "not_established"
                    and r[method]["decision"] in ["supported", "contradicted"]
                    for r in rr
                ),
            )
            for method in METHODS
        }
        out[m]["profile_exact"] = sum(r["profile_exact"] for r in rr)
        pp = {r["bundle"]: r["policy_equivalent"] for r in rr}
        out[m]["policy_equivalent"] = sum(v is True for v in pp.values())
        out[m]["policy_n"] = len(pp)
        out[m]["by_bundle"] = {
            b: {
                method: sum(
                    r[method]["decision"] == r["label"] for r in rr if r["bundle"] == b
                )
                for method in METHODS
            }
            for b in pp
        }
    return out


def run(inputs, cache, dest, method):
    data = json.loads(Path(inputs).read_text())
    cache = Path(cache)
    dest = Path(dest)
    bs = {b["id"]: b for b in data["bundles"]}
    records = []
    rows = []
    reqfn = extended_request if method == "extended" else base_request
    for model in MODELS:
        if not (cache / model).exists():
            continue
        raw = {}
        jobs = [("policy", b["id"], b, None) for b in data["bundles"]] + [
            (s, c["id"], bs[c["bundle"]], c)
            for c in data["cases"]
            for s in ["profile", "direct"]
        ]
        for stage, key, b, c in jobs:
            record = json.loads((cache / model / stage / (key + ".json")).read_text())
            req = reqfn(b, c, model, stage)
            assert record["request_sha256"] == sha(req) and record[
                "input_sha256"
            ] == sha(req["messages"])
            raw[stage, key] = record["parsed"]
            meta = {
                k: record[k]
                for k in [
                    "id",
                    "stage",
                    "model",
                    "model_name",
                    "model_digest",
                    "server",
                    "created_utc",
                    "request_sha256",
                    "input_sha256",
                    "elapsed_seconds",
                ]
            }
            meta.update(
                parsed=record["parsed"],
                response_sha256=sha(record["response"]),
                done_reason=record["response"].get("done_reason"),
                reported_eval_count=record["response"].get("eval_count"),
            )
            records.append(meta)
        for c in data["cases"]:
            b = bs[c["bundle"]]
            allowed = list(b["evidence"])
            p = adapt(raw["policy", b["id"]])
            q = raw["profile", c["id"]]
            rp = b["reference_policy"]
            rq = c["reference_profile"]
            row = dict(
                id=c["id"],
                bundle=b["id"],
                model=model,
                label=c["label"],
                direct={"decision": valid_direct(raw["direct", c["id"]], allowed)},
                pipeline=interpret(p, q, allowed),
                reference_policy=interpret(rp, q, allowed),
                reference_profile=interpret(p, rq, allowed),
                both_reference=interpret(rp, rq, allowed),
                profile_exact=profile_exact(q, rq),
                policy_equivalent=equivalent(p, rp, allowed),
            )
            assert row["both_reference"]["decision"] == c["label"], row
            rows.append(row)
    dest.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("predictions", records),
        ("rows", rows),
        ("summary", summarize(rows)),
    ]:
        (dest / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    print(
        json.dumps(
            {"rows": len(rows), "responses": len(records), "summary": summarize(rows)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--method", choices=["base", "extended"], required=True)
    a = p.parse_args()
    run(a.inputs, a.cache, a.dest, a.method)
