"""Replay diagnostics: source coverage, real cache errors, routing overhead and stress."""

import argparse
import copy
import json
from pathlib import Path
from core import decision
from evaluate import score
from infer import ROOT
from prepare_inputs import public_stub
from provenance import evaluate as coverage
from report import export
from routing import GATES


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    (ROOT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def calculate(split):
    ref = read(
        "development.json" if split == "development" else split + "_reference.json"
    )
    data = ref if split == "development" else public_stub(ref)
    scores = read("results/" + split + "/case_scores.json")
    result = {}
    for model, methods in scores.items():
        records = read("results/" + split + "/" + model + ".json")
        runtime = {}
        for method, v in methods.items():
            cases = v["cases"]
            calls = sum(r["refreshed"] for r in cases)
            elapsed = sum(
                records["after"][r["id"]]["elapsed_seconds"]
                for r in cases
                if r["refreshed"]
            )
            if method in GATES:
                calls += len(cases)
                elapsed += sum(
                    r["elapsed_seconds"] for r in records[GATES[method]].values()
                )
            if method in ["change_map_gate", "audited_change_map"]:
                calls += len(records["map"])
                elapsed += sum(r["elapsed_seconds"] for r in records["map"].values())
            runtime[method] = {
                "counterfactual_update_calls": calls,
                "sum_observed_stage_seconds": round(elapsed, 3),
                "fresh_reads_avoided": len(cases) - v["metrics"]["refreshed"],
            }
        rows = methods["refresh_all"]["cases"]
        oracle = []
        for r in rows:
            recheck = r["required_recheck"]
            final = r["fresh"] if recheck else r["old"]
            oracle.append(
                {
                    "id": r["id"],
                    "refreshed": recheck,
                    "correct": final["decision"] == r["reference"]["after"],
                    "retained_old_error": not r["old_correct"]
                    and not recheck
                    and final["decision"] != r["reference"]["after"],
                }
            )
        corrupt = copy.deepcopy(records)
        cycle = {
            "supported": "contradicted",
            "contradicted": "not_established",
            "not_established": "supported",
        }
        changed = []
        for key, r in corrupt["before"].items():
            if isinstance(r["parsed"], dict) and r["parsed"].get("decision") in cycle:
                r["parsed"]["decision"] = cycle[r["parsed"]["decision"]]
                changed.append(key)
        # Gates only receive old evidence IDs, which this intervention preserves.
        stress = score(data, corrupt)
        result[model] = {
            "cost": runtime,
            "oracle_route_diagnostic": {
                "cases": oracle,
                "correct": sum(r["correct"] for r in oracle),
                "refreshed": sum(r["refreshed"] for r in oracle),
                "retained_old_error": sum(r["retained_old_error"] for r in oracle),
            },
            "corrupted_cache_diagnostic": {
                "changed_ids": changed,
                "rule": "Rotate valid semantic labels S->C->U->S, preserve evidence IDs, reuse identical gate requests. Artificial stress, not naturally occurring errors.",
                "metrics": {k: v["metrics"] for k, v in stress.items()},
            },
            "correct_cache_strata": {
                method: {
                    str(old_ok): {
                        "n": sum(r["old_correct"] == old_ok for r in v["cases"]),
                        "correct": sum(
                            r["old_correct"] == old_ok and r["correct"]
                            for r in v["cases"]
                        ),
                        "refreshed": sum(
                            r["old_correct"] == old_ok and r["refreshed"]
                            for r in v["cases"]
                        ),
                    }
                    for old_ok in [True, False]
                }
                for method, v in methods.items()
            },
        }
    return result


def repairs(cache):
    result = {}
    references = {
        s: read(s + "_reference.json") for s in ["retrieved", "top5", "neighbor5"]
    }
    for split, ref in references.items():
        bs = {b["id"]: b for b in ref["bundles"]}
        models = {}
        for model in ["qwen", "kanana-public"]:
            if split == "retrieved":
                records = read("results/retrieved/" + model + ".json")["after"]
            else:
                records = {
                    p.stem: export(json.loads(p.read_text()))
                    for p in (Path(cache) / split / model).glob("*.json")
                }
                assert len(records) == 32
                folder = ROOT / "results" / split
                folder.mkdir(exist_ok=True)
                write("results/" + split + "/" + model + ".json", records)
            rows = []
            for c in ref["cases"]:
                b = bs[c["bundle"]]
                loc = b["source_locators"]
                allowed = {**loc["before"], **loc["after"]}
                pred = decision(records[c["id"]]["parsed"], allowed)
                from provenance import covered

                support = covered(
                    loc["after"], pred["evidence"], c["required_source_spans"]["after"]
                )
                rows.append(
                    {
                        "id": c["id"],
                        "reference": c["reference"]["after"],
                        "prediction": pred,
                        "correct": pred["decision"] == c["reference"]["after"],
                        "cited_full_span": support,
                        "joint": pred["decision"] == c["reference"]["after"]
                        and support,
                        "supplied_full_span": c["retrieval_coverage"]["after"],
                        "elapsed_seconds": records[c["id"]]["elapsed_seconds"],
                        "input_characters": sum(
                            l["end"] - l["start"]
                            for v in loc.values()
                            for l in v.values()
                        ),
                    }
                )
            models[model] = {
                "cases": rows,
                "metrics": {
                    k: sum(r[k] for r in rows)
                    for k in [
                        "correct",
                        "cited_full_span",
                        "joint",
                        "supplied_full_span",
                    ]
                },
                "sum_observed_seconds": round(
                    sum(r["elapsed_seconds"] for r in rows), 3
                ),
                "n": len(rows),
            }
        result[split] = models
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    a = p.parse_args()
    for split in ["development", "known", "curated", "retrieved"]:
        if (ROOT / "results" / split / "summary.json").exists():
            write("results/" + split + "/diagnostics.json", calculate(split))
            if split in ["curated", "retrieved"]:
                write(
                    "results/" + split + "/provenance.json",
                    coverage(
                        read(split + "_reference.json"),
                        read("results/" + split + "/case_scores.json"),
                    ),
                )
    if a.cache:
        write("results/retrieval_repairs.json", repairs(a.cache))
