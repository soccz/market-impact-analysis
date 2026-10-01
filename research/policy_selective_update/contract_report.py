"""Publish numeric contracts and compare query-independent execution to matched reads."""

import argparse
import json
from pathlib import Path
from core import decision
from infer import ROOT
from report import export
from temporal_contract import validate, predict


def calculate(records):
    boundary = json.loads((ROOT / "contract_reference.json").read_text())
    ref = json.loads((ROOT / "curated_reference.json").read_text())
    cases = [c for c in ref["cases"] if c["id"].startswith("seoul-cap-")]
    result = {}
    for model, stages in records.items():
        compiled = {
            phase: validate(
                stages["contract"][phase]["parsed"],
                ["OCAP" if phase == "before" else "NCAP"],
            )
            for phase in ["before", "after"]
        }
        rows = []
        for c in boundary:
            predictions = {
                p: predict(compiled[p], c["joined_on"], c["asserted_cap"])
                for p in ["before", "after"]
            }
            rows.append(
                {
                    **c,
                    "predictions": predictions,
                    "correct": {p: predictions[p] == c[p] for p in predictions},
                }
            )
        matched = []
        for c in cases:
            key = c["id"]
            cap = 400000 if key.endswith("-40") else 300000
            day = (
                "2025-04-01"
                if "-new-" in key
                else "2025-03-30" if "-old-" in key else None
            )
            direct = decision(stages["matched"][key]["parsed"], ["OCAP", "NCAP"])
            answer = predict(compiled["after"], day, cap)
            matched.append(
                {
                    "id": key,
                    "reference": c["reference"]["after"],
                    "direct": direct,
                    "contract_prediction": answer,
                    "direct_correct": direct["decision"] == c["reference"]["after"],
                    "contract_correct": answer == c["reference"]["after"],
                }
            )
        result[model] = {
            "contracts_valid": {p: compiled[p] is not None for p in compiled},
            "boundary_cases": rows,
            "matched_cases": matched,
            "summary": {
                "boundary_n": len(rows),
                "before_correct": sum(c["correct"]["before"] for c in rows),
                "after_correct": sum(c["correct"]["after"] for c in rows),
                "matched_n": len(matched),
                "direct_correct": sum(c["direct_correct"] for c in matched),
                "contract_correct": sum(c["contract_correct"] for c in matched),
            },
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    records = {}
    for model in ["qwen", "kanana-public"]:
        records[model] = {
            stage: {
                r.stem: export(json.loads(r.read_text()))
                for r in (Path(a.cache) / "contract" / model / stage).glob("*.json")
            }
            for stage in ["contract", "matched"]
        }
        assert (
            len(records[model]["contract"]) == 2 and len(records[model]["matched"]) == 6
        )
    out = ROOT / "results/contract"
    out.mkdir(exist_ok=True)
    for name, value in [("predictions", records), ("summary", calculate(records))]:
        (out / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print({m: r["summary"] for m, r in calculate(records).items()})
