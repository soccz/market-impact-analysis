"""Scoring and provenance for frozen source transfer; unavailable != unknown."""

import argparse
import json
from pathlib import Path
from bridge import ROOT, MODELS, sha, profile_exact, adapt
from candidates import extract, rule_selection, compile_selection, public_catalog
from source_method import request, compile_response, coverage
from claim_alignment import align
from evaluate_v2 import execute
from extended_logic import equivalent

METHODS = ["direct", "rules_raw_facts", "rules", "model_raw_facts", "model", "freeform"]


def summary(rows):
    out = {}
    for model in MODELS:
        rr = [r for r in rows if r["model_id"] == model]
        out[model] = {}
        for method in METHODS:
            if not all(method in r for r in rr):
                continue
            out[model][method] = dict(
                n=len(rr),
                correct=sum(r[method] == r["label"] for r in rr),
                abstain=sum(r[method] == "abstain" for r in rr),
            )
            if method != "direct":
                key = (
                    "raw_profile_exact"
                    if method.endswith("raw_facts")
                    else "profile_exact"
                )
                out[model][method]["joint"] = sum(
                    r[method] == r["label"] and r[key] for r in rr
                )
        out[model]["facts"] = dict(
            raw=sum(r["raw_profile_exact"] for r in rr),
            aligned=sum(r["profile_exact"] for r in rr),
            n=len(rr),
        )
    return out


def public_record(raw):
    r = {
        k: raw[k]
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
            "parsed",
        ]
    }
    r["response_sha256"] = sha(raw["response"])
    return r


def export(
    data,
    cache,
    dest,
    extract_fn=extract,
    compile_fn=compile_response,
    coverage_fn=coverage,
    request_fn=request,
    policy_cache=None,
    freeform_cache=None,
):
    records = []
    lookup = {}
    bs = {b["id"]: b for b in data["bundles"]}
    for model in MODELS:
        jobs = [("policy", b["id"], b, None) for b in data["bundles"]] + [
            (s, c["id"], bs[c["bundle"]], c)
            for c in data["cases"]
            for s in ["profile", "direct"]
        ]
        for stage, key, b, c in jobs:
            base = (
                Path(policy_cache)
                if stage == "policy" and policy_cache
                else Path(cache)
            )
            raw = json.loads((base / model / stage / (key + ".json")).read_text())
            req = request_fn(b, c, model, stage)
            assert raw["request_sha256"] == sha(req) and raw["input_sha256"] == sha(
                req["messages"]
            )
            r = public_record(raw)
            records.append(r)
            lookup[model, stage, key] = r["parsed"]
    rows = []
    rules = []
    for model in MODELS:
        for b in data["bundles"]:
            catalog = extract_fn(b)
            allowed = list(b["evidence"])
            rule = (
                compile_selection(rule_selection(b, catalog), catalog)
                if coverage_fn(b)["covered"]
                else dict(status="insufficient", clauses=[], derived=[])
            )
            policy = compile_fn(b, lookup[model, "policy", b["id"]])
            r = dict(
                model_id=model,
                bundle=b["id"],
                allowed=allowed,
                catalog=public_catalog(catalog),
                coverage=coverage_fn(b),
                rule_policy=rule,
                model_policy=policy,
                reference_policy=adapt(b["reference_policy"]),
                rule_equivalent=equivalent(rule, adapt(b["reference_policy"]), allowed),
                model_equivalent=equivalent(
                    policy, adapt(b["reference_policy"]), allowed
                ),
            )
            if freeform_cache:
                raw = json.loads(
                    (
                        Path(freeform_cache) / model / "policy" / (b["id"] + ".json")
                    ).read_text()
                )
                r["freeform_policy"] = raw["parsed"]
                records.append(public_record(raw))
                from bridge import old_request

                assert raw["request_sha256"] == sha(
                    old_request(b, None, model, "policy")
                )
                r["freeform_equivalent"] = equivalent(
                    raw["parsed"], adapt(b["reference_policy"]), allowed
                )
            rules.append(r)
        for c in data["cases"]:
            b = bs[c["bundle"]]
            rr = next(
                r for r in rules if r["model_id"] == model and r["bundle"] == b["id"]
            )
            raw = lookup[model, "profile", c["id"]]
            facts, audit = align(c["claim"], raw)
            allowed = rr["allowed"]
            direct = lookup[model, "direct", c["id"]]
            row = dict(
                model_id=model,
                id=c["id"],
                bundle=b["id"],
                label=c["label"],
                claim=c["claim"],
                raw_profile=raw,
                profile=facts,
                alignment=audit,
                raw_profile_exact=profile_exact(raw, c["reference_profile"]),
                profile_exact=profile_exact(facts, c["reference_profile"]),
                direct=(
                    direct.get("decision", "abstain")
                    if isinstance(direct, dict)
                    else "abstain"
                ),
                rules_raw_facts=execute(rr["rule_policy"], raw, allowed),
                rules=execute(rr["rule_policy"], facts, allowed),
                model_raw_facts=execute(rr["model_policy"], raw, allowed),
                model=execute(rr["model_policy"], facts, allowed),
            )
            if freeform_cache:
                row["freeform"] = execute(rr["freeform_policy"], facts, allowed)
            rows.append(row)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    result = dict(predictions=records, rows=rows, rules=rules, summary=summary(rows))
    for name, value in result.items():
        (dest / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--freeform-cache")
    a = p.parse_args()
    print(
        json.dumps(
            export(
                json.loads(Path(a.inputs).read_text()),
                a.cache,
                a.dest,
                freeform_cache=a.freeform_cache,
            )["summary"],
            ensure_ascii=False,
        )
    )
