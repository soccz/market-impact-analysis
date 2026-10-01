"""Recompute first-response comparisons and error localization without inference."""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
from logic import policy, profile, interpret, evaluate
from worlds import candidates
from infer import MODELS, request, sha

ROOT = Path(__file__).parent


def equivalent(pred, reference, allowed):
    p, ps = policy(pred, allowed)
    r, rs = policy(reference, allowed)
    if p is None or r is None:
        return {
            "equivalent": ps == rs == "policy_insufficient",
            "worlds": 0,
            "status": f"{ps}/{rs}",
        }
    clauses = p + r
    fields = sorted({a["field"] for c in clauses for a in c})
    domains = [candidates(f, clauses) for f in fields]
    count = 1
    for d in domains:
        count *= len(d)
    if count > 100000:
        return {"equivalent": None, "worlds": count, "status": "comparison_budget"}
    mismatches = 0
    for v in itertools.product(*domains):
        vals = dict(zip(fields, v))
        mismatches += evaluate(p, vals)[0] != evaluate(r, vals)[0]
    return {
        "equivalent": mismatches == 0,
        "worlds": count,
        "mismatches": mismatches,
        "status": "checked",
    }


def profile_equal(pred, reference):
    p, pa, ps = profile(pred)
    r, ra, rs = profile(reference)
    if ps != rs:
        return False
    if ps == "query_unsupported":
        return True
    if ps != "valid":
        return False
    return pa == ra and {k: v for k, v in p.items() if v is not None} == {
        k: v for k, v in r.items() if v is not None
    }


def valid_direct(parsed, allowed):
    if (
        not isinstance(parsed, dict)
        or parsed.get("decision")
        not in ["supported", "contradicted", "not_established"]
        or not isinstance(parsed.get("evidence"), list)
        or any(x not in allowed for x in parsed["evidence"])
    ):
        return "abstain"
    return parsed["decision"]


def summarize(rows):
    methods = [
        "direct",
        "pipeline",
        "kleene",
        "reference_policy",
        "reference_profile",
        "both_reference",
    ]
    out = {}
    for model in MODELS:
        rr = [r for r in rows if r["model"] == model]
        if not rr:
            continue
        out[model] = {}
        for method in methods:
            correct = sum(r[method]["decision"] == r["label"] for r in rr)
            n = len(rr)
            confusion = Counter((r["label"], r[method]["decision"]) for r in rr)
            out[model][method] = dict(
                n=n,
                correct=correct,
                accuracy=correct / n,
                abstain=sum(r[method]["decision"] == "abstain" for r in rr),
                unknown=sum(r[method]["decision"] == "not_established" for r in rr),
                false_determination=sum(
                    r["label"] == "not_established"
                    and r[method]["decision"] in ["supported", "contradicted"]
                    for r in rr
                ),
                confusion={f"{x}/{y}": v for (x, y), v in sorted(confusion.items())},
            )
        out[model]["profile_exact"] = sum(r["profile_exact"] for r in rr)
        pr = {r["bundle"]: r["policy_equivalence"] for r in rr}
        out[model]["policy_equivalent"] = sum(
            v["equivalent"] is True for v in pr.values()
        )
        out[model]["policy_n"] = len(pr)
        out[model]["finite_changed_cases"] = [
            r["id"] for r in rr if r["pipeline"]["decision"] != r["kleene"]["decision"]
        ]
        out[model]["pipeline_fixes_direct"] = [
            r["id"]
            for r in rr
            if r["pipeline"]["decision"] == r["label"]
            and r["direct"]["decision"] != r["label"]
        ]
        out[model]["pipeline_breaks_direct"] = [
            r["id"]
            for r in rr
            if r["pipeline"]["decision"] != r["label"]
            and r["direct"]["decision"] == r["label"]
        ]
    return out


def run(inputs, cache, dest, method="baseline"):
    if method == "explicit":
        from infer_explicit import request as build_request
    else:
        build_request = request
    data = json.loads(Path(inputs).read_text())
    cache = Path(cache)
    dest = Path(dest)
    bundles = {b["id"]: b for b in data["bundles"]}
    rows = []
    records = []
    for model in MODELS:
        if not (cache / model).exists():
            continue
        raw = {}
        jobs = [("policy", b["id"], b, None) for b in data["bundles"]] + [
            (s, c["id"], bundles[c["bundle"]], c)
            for c in data["cases"]
            for s in ["profile", "direct"]
        ]
        for stage, key, b, c in jobs:
            r = json.loads((cache / model / stage / (key + ".json")).read_text())
            req = build_request(b, c, model, stage)
            assert r["request_sha256"] == sha(req), (model, stage, key, "request")
            assert r["input_sha256"] == sha(req["messages"])
            raw[stage, key] = r["parsed"]
            meta = {
                k: r[k]
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
                parsed=r["parsed"],
                response_sha256=sha(r["response"]),
                done_reason=r["response"].get("done_reason"),
                reported_eval_count=r["response"].get("eval_count"),
            )
            records.append(meta)
        for c in data["cases"]:
            b = bundles[c["bundle"]]
            p = raw["policy", b["id"]]
            q = raw["profile", c["id"]]
            allowed = list(b["evidence"])
            rp = b["reference_policy"]
            rq = c["reference_profile"]
            row = dict(
                id=c["id"],
                bundle=b["id"],
                model=model,
                label=c["label"],
                direct={"decision": valid_direct(raw["direct", c["id"]], allowed)},
                pipeline=interpret(p, q, allowed),
                kleene=interpret(p, q, allowed, "kleene"),
                reference_policy=interpret(rp, q, allowed),
                reference_profile=interpret(p, rq, allowed),
                both_reference=interpret(rp, rq, allowed),
                profile_exact=profile_equal(q, rq),
                policy_equivalence=equivalent(p, rp, allowed),
            )
            assert row["both_reference"]["decision"] == c["label"], row
            rows.append(row)
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in [
        ("rows", rows),
        ("predictions", records),
        ("summary", summarize(rows)),
    ]:
        (dest / (name + ".json")).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
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
    p.add_argument("--method", choices=["baseline", "explicit"], default="baseline")
    a = p.parse_args()
    run(a.inputs, a.cache, a.dest, a.method)
