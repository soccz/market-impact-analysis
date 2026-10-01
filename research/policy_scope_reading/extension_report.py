"""Keep post-hoc reuse apart from the newly held-out institution."""

import argparse
import json
from pathlib import Path

from evaluate import parsed
from infer import ROOT
from report import enriched, export_official

METHODS = [
    "kanana-public-baseline",
    "kanana-public-evidence",
    "qwen-baseline",
    "qwen-evidence",
]


def run(cache, output=None):
    cache = Path(cache)
    out = Path(output or ROOT / "results/extension")
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    diagnostics = {}
    for split in ["development", "official", "replication"]:
        rows = json.loads(
            (
                ROOT / "development.json"
                if split == "development"
                else cache / (split + "_inputs.json")
            ).read_text()
        )
        predictions = {}
        raw = {}
        for name in METHODS:
            if name == "qwen-baseline" and split != "replication":
                folder = (
                    ROOT / "results/development" / name
                    if split == "development"
                    else cache / "results/official" / name
                )
            else:
                folder = (
                    (
                        ROOT / "results/extension"
                        if split == "development"
                        else cache / "results/extension"
                    )
                    / split
                    / name
                )
            raw[name] = {
                r["id"]: json.loads((folder / (r["id"] + ".json")).read_text())
                for r in rows
            }
            predictions[name] = {r["id"]: parsed(raw[name][r["id"]], r) for r in rows}
        if split != "development":
            export_official(raw, out / split)
            for name, records in raw.items():
                for cid, record in records.items():
                    if (
                        isinstance(record.get("parsed"), dict)
                        and "coverage" in record["parsed"]
                    ):
                        path = out / split / name / (cid + ".json")
                        pub = json.loads(path.read_text())
                        pub["parsed"]["coverage"] = record["parsed"]["coverage"]
                        path.write_text(
                            json.dumps(pub, ensure_ascii=False, indent=2) + "\n"
                        )
            base = [r for r in rows if "variant" not in r]
            pairs = {}
            for name, pred in predictions.items():
                pairs[name] = dict(
                    pairs=len(base),
                    both_base_and_distractor_correct=sum(
                        pred[r["id"]]["decision"]
                        == r["reference"]["decision"]
                        == pred[r["id"] + "-distractor"]["decision"]
                        for r in base
                    ),
                    removed_semantic_unknown=sum(
                        pred[r["id"] + "-removed"]["decision"] == "not_established"
                        for r in base
                    ),
                    coverage_decision_conflicts=sum(
                        isinstance(v.get("parsed"), dict)
                        and v["parsed"].get("coverage") == "insufficient"
                        and v["parsed"].get("decision") in {"supported", "contradicted"}
                        for v in raw[name].values()
                    ),
                )
            diagnostics[split] = pairs
        for variant in (
            [None] if split == "development" else ["base", "distractor", "removed"]
        ):
            selected = (
                rows
                if variant is None
                else [r for r in rows if r.get("variant", "base") == variant]
            )
            support = (
                json.loads((ROOT / "development_support.json").read_text())[
                    "support_sets"
                ]
                if split == "development"
                else None
            )
            key = split if variant is None else split + "_" + variant
            groups[key] = enriched(selected, predictions, support)
    pairs = {}
    for group, results in groups.items():
        pairs[group] = {}
        for model in ["kanana-public", "qwen"]:
            a = {r["id"]: r for r in results[model + "-baseline"]["cases"]}
            b = {r["id"]: r for r in results[model + "-evidence"]["cases"]}
            pairs[group][model] = dict(
                n=len(a),
                gained=sum(not a[k]["correct"] and b[k]["correct"] for k in a),
                lost=sum(a[k]["correct"] and not b[k]["correct"] for k in a),
                both_correct=sum(a[k]["correct"] and b[k]["correct"] for k in a),
            )
    summary = {k: {m: s["metrics"] for m, s in v.items()} for k, v in groups.items()}
    for name, data in [
        ("summary.json", summary),
        ("case_scores.json", groups),
        ("diagnostics.json", diagnostics),
        ("paired_changes.json", pairs),
    ]:
        (out / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                g: {
                    m: {
                        k: s[k]
                        for k in ["correct", "joint", "wrong_definitive", "abstain"]
                    }
                    for m, s in v.items()
                }
                for g, v in summary.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    run(a.cache, a.output)
