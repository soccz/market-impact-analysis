"""Post-hoc narrower source spans; original labels and primary scores stay fixed."""

import json
from core import decision
from infer import ROOT
from provenance import covered


def calculate():
    reference = json.loads((ROOT / "support_sensitivity.json").read_text())[
        "after_support"
    ]
    out = {}
    for split in ["curated", "retrieved", "top5", "neighbor5"]:
        ref = json.loads((ROOT / (split + "_reference.json")).read_text())
        bs = {b["id"]: b for b in ref["bundles"]}
        out[split] = {}
        for model in ["qwen", "kanana-public"]:
            records = json.loads(
                (ROOT / "results" / split / (model + ".json")).read_text()
            )
            if split in ["curated", "retrieved"]:
                records = records["after"]
            rows = []
            for c in ref["cases"]:
                locations = bs[c["bundle"]]["source_locators"]
                loc = locations["after"]
                required = [reference[c["id"]]]
                pred = decision(
                    records[c["id"]]["parsed"], {**locations["before"], **loc}
                )
                supplied = covered(loc, list(loc), required)
                cited = covered(loc, pred["evidence"], required)
                correct = pred["decision"] == c["reference"]["after"]
                rows.append(
                    {
                        "id": c["id"],
                        "supplied_narrow_span": supplied,
                        "cited_narrow_span": cited,
                        "correct": correct,
                        "narrow_joint": cited and correct,
                    }
                )
            out[split][model] = {
                "cases": rows,
                "metrics": {
                    k: sum(r[k] for r in rows)
                    for k in [
                        "supplied_narrow_span",
                        "cited_narrow_span",
                        "correct",
                        "narrow_joint",
                    ]
                },
            }
    return out


if __name__ == "__main__":
    out = calculate()
    (ROOT / "results/support_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        {s: {m: v["metrics"] for m, v in models.items()} for s, models in out.items()}
    )
