"""Same saved responses scored before rejection and conservative correction."""

import argparse, json
from pathlib import Path
from collections import Counter
from bridge import (
    ROOT,
    MODELS,
    request,
    sha,
    interpret,
    profile_exact,
    valid_direct,
    adapt,
)
from grounding import audit, correct


def score(data, records):
    bs = {b["id"]: b for b in data["bundles"]}
    look = {(r["model"], r["stage"], r["id"]): r["parsed"] for r in records}
    rows = []
    for model in MODELS:
        for c in data["cases"]:
            b = bs[c["bundle"]]
            allowed = list(b["evidence"])
            p = adapt(look[model, "policy", b["id"]])
            q = look[model, "profile", c["id"]]
            a = audit(c["claim"], q)
            fixed = correct(q, a)
            raw = interpret(p, q, allowed)
            patched = interpret(p, fixed, allowed)
            refq = c["reference_profile"]
            assert (
                interpret(b["reference_policy"], refq, allowed)["decision"]
                == c["label"]
            )
            rows.append(
                dict(
                    id=c["id"],
                    bundle=c["bundle"],
                    model=model,
                    label=c["label"],
                    group=c.get("group"),
                    variant=c.get("variant"),
                    direct=valid_direct(look[model, "direct", c["id"]], allowed),
                    raw=raw["decision"],
                    reject="abstain" if a["issues"] else raw["decision"],
                    corrected=patched["decision"],
                    raw_profile_exact=profile_exact(q, refq),
                    corrected_profile_exact=profile_exact(fixed, refq),
                    audit=a,
                    corrected_profile=fixed,
                    reference_rule=interpret(b["reference_policy"], fixed, allowed)[
                        "decision"
                    ],
                )
            )
    summary = {}
    pairs = []
    for model in MODELS:
        rr = [r for r in rows if r["model"] == model]
        summary[model] = {}
        for method in ["direct", "raw", "reject", "corrected", "reference_rule"]:
            summary[model][method] = dict(
                n=len(rr),
                correct=sum(r[method] == r["label"] for r in rr),
                abstain=sum(r[method] == "abstain" for r in rr),
                unknown=sum(r[method] == "not_established" for r in rr),
                confusion=dict(
                    sorted(Counter(r["label"] + "/" + r[method] for r in rr).items())
                ),
            )
        for k in ["raw_profile_exact", "corrected_profile_exact"]:
            summary[model][k] = sum(r[k] for r in rr)
        summary[model].update(
            grounding_covered=sum(bool(r["audit"]["checked"]) for r in rr),
            flagged=sum(bool(r["audit"]["issues"]) for r in rr),
            joint_raw=sum(
                r["raw"] == r["label"] and r["raw_profile_exact"] for r in rr
            ),
            joint_corrected=sum(
                r["corrected"] == r["label"] and r["corrected_profile_exact"]
                for r in rr
            ),
            answer_fixes=sum(
                r["raw"] != r["label"] and r["corrected"] == r["label"] for r in rr
            ),
            answer_harms=sum(
                r["raw"] == r["label"] and r["corrected"] != r["label"] for r in rr
            ),
            profile_fixes=sum(
                not r["raw_profile_exact"] and r["corrected_profile_exact"] for r in rr
            ),
            profile_harms=sum(
                r["raw_profile_exact"] and not r["corrected_profile_exact"] for r in rr
            ),
        )
        for group in sorted({r["group"] for r in rr if r["group"]}):
            gg = {r["variant"]: r for r in rr if r["group"] == group}
            base = gg["base"]
            for v, other in gg.items():
                if v == "base":
                    continue
                expected_same = base["label"] == other["label"]
                pairs.append(
                    dict(
                        model=model,
                        group=group,
                        variant=v,
                        expected_same=expected_same,
                        methods={
                            method: dict(
                                relation_correct=(base[method] == other[method])
                                == expected_same,
                                both_correct=base[method] == base["label"]
                                and other[method] == other["label"],
                                base=base[method],
                                variant=other[method],
                            )
                            for method in ["direct", "raw", "reject", "corrected"]
                        },
                    )
                )
    context_pairs = []
    for model in MODELS:
        lookup = {r["id"]: r for r in rows if r["model"] == model}
        for pair in data.get("context_pairs", []):
            left, right = lookup[pair["left"]], lookup[pair["right"]]
            context_pairs.append(
                dict(
                    model=model,
                    **pair,
                    methods={
                        m: dict(
                            changed=left[m] != right[m],
                            both_correct=left[m] == left["label"]
                            and right[m] == right["label"],
                        )
                        for m in ["direct", "raw", "reject", "corrected"]
                    }
                )
            )
    return dict(rows=rows, summary=summary, pairs=pairs, context_pairs=context_pairs)


def export(inputs, cache, dest):
    data = json.loads(Path(inputs).read_text())
    cache = Path(cache)
    bs = {b["id"]: b for b in data["bundles"]}
    records = []
    for model in MODELS:
        jobs = [("policy", b["id"], b, None) for b in data["bundles"]] + [
            (s, c["id"], bs[c["bundle"]], c)
            for c in data["cases"]
            for s in ["profile", "direct"]
        ]
        for stage, key, b, c in jobs:
            r = json.loads((cache / model / stage / (key + ".json")).read_text())
            req = request(b, c, model, stage)
            assert r["request_sha256"] == sha(req) and r["input_sha256"] == sha(
                req["messages"]
            )
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
                    "parsed",
                ]
            }
            meta.update(
                response_sha256=sha(r["response"]),
                done_reason=r["response"].get("done_reason"),
            )
            records.append(meta)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, value in dict(predictions=records, **score(data, records)).items():
        (dest / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(score(data, records)["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--dest", required=True)
    a = p.parse_args()
    export(a.inputs, a.cache, a.dest)
