"""Active evaluator: fixes the request-builder injection hook; scoring unchanged."""

import json
from pathlib import Path
from bridge import ROOT, FACT, MODELS, adapt, interpret, profile_exact, sha, old_request
from candidates import extract, compile_selection, rule_selection, public_catalog
from infer_candidates import request
from extended_logic import equivalent


def execute(policy, profile, allowed):
    if not isinstance(policy, dict) or policy.get("status") != "ready":
        return "abstain"
    return interpret(policy, profile, allowed)["decision"]


def development_data(private_root):
    data = dict(bundles=[], cases=[])
    for split, path in [
        ("metamorphic", FACT / "metamorphic.json"),
        ("official", Path(private_root) / "official_inputs.json"),
    ]:
        ds = json.loads(path.read_text())
        for c in ds["cases"]:
            c["dataset"] = split
        data["bundles"] += ds["bundles"]
        data["cases"] += ds["cases"]
    return data


def previous_records():
    return [
        r
        for split in ["metamorphic", "official"]
        for r in json.loads((FACT / f"results/{split}/predictions.json").read_text())
    ]


def previous_rows():
    return [
        r
        for split in ["metamorphic", "official"]
        for r in json.loads((FACT / f"results/{split}/rows.json").read_text())
    ]


def evaluate(data, model_records, baseline_records, baseline_rows):
    bs = {b["id"]: b for b in data["bundles"]}
    new = {(r["model"], r["id"]): r for r in model_records}
    old = {(r["model"], r["id"]): r for r in baseline_rows}
    rows = []
    rules = []
    for m in MODELS:
        for b in data["bundles"]:
            catalog = extract(b)
            rule = compile_selection(rule_selection(b, catalog), catalog)
            p = compile_selection(new[m, b["id"]]["parsed"], catalog)
            rp = adapt(b["reference_policy"])
            allowed = list(b["evidence"])
            rules.append(
                dict(
                    model=m,
                    bundle=b["id"],
                    catalog=public_catalog(catalog),
                    rule_policy=rule,
                    model_policy=p,
                    rule_equivalent=equivalent(rule, rp, allowed),
                    model_equivalent=equivalent(p, rp, allowed),
                )
            )
        for c in data["cases"]:
            b = bs[c["bundle"]]
            prior = old[m, c["id"]]
            q = prior["corrected_profile"]
            r = next(r for r in rules if r["model"] == m and r["bundle"] == b["id"])
            allowed = list(b["evidence"])
            row = dict(
                id=c["id"],
                bundle=b["id"],
                model=m,
                dataset=c.get("dataset", "official"),
                label=c["label"],
                profile=q,
                profile_exact=profile_exact(q, c["reference_profile"]),
                direct=prior["direct"],
                previous=prior["corrected"],
                candidate_rules=execute(r["rule_policy"], q, allowed),
                candidate_model=execute(r["model_policy"], q, allowed),
            )
            rows.append(row)
    summary = {}
    for ds in sorted({r["dataset"] for r in rows}):
        summary[ds] = {}
        for m in MODELS:
            rr = [r for r in rows if r["model"] == m and r["dataset"] == ds]
            summary[ds][m] = {}
            for method in ["direct", "previous", "candidate_rules", "candidate_model"]:
                summary[ds][m][method] = dict(
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
    return dict(rows=rows, summary=summary, rules=rules)


def export(
    data,
    cache,
    dest,
    baseline_records=None,
    baseline_rows=None,
    feedbacks=None,
    request_fn=request,
):
    records = []
    for m in MODELS:
        for b in data["bundles"]:
            raw = json.loads(
                (Path(cache) / m / "policy" / (b["id"] + ".json")).read_text()
            )
            req = request_fn(
                b,
                None,
                m,
                "policy",
                None if feedbacks is None else feedbacks.get((m, b["id"])),
            )
            assert raw["request_sha256"] == sha(req) and raw["input_sha256"] == sha(
                req["messages"]
            )
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
            records.append(r)
    result = evaluate(
        data,
        records,
        baseline_records or previous_records(),
        baseline_rows or previous_rows(),
    )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, value in dict(predictions=records, **result).items():
        (dest / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(result["summary"], ensure_ascii=False))
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--previous-cache", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--dest", required=True)
    a = p.parse_args()
    export(development_data(a.previous_cache), a.cache, a.dest)
