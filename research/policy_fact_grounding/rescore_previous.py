"""Post-hoc reanalysis of frozen responses; no additional model inference."""

import json
from pathlib import Path
from bridge import ROOT, PREVIOUS
from report import score

SPECS = {
    "official": "official_reference.json",
    "composition": "composition.json",
    "capability_extended": "capability_reference.json",
}


def calculate(split):
    data = json.loads((PREVIOUS / SPECS[split]).read_text())
    for b in data["bundles"]:
        b["reference_policy"].setdefault("derived", [])
    records = json.loads((PREVIOUS / f"results/{split}/predictions.json").read_text())
    return score(data, records)


if __name__ == "__main__":
    for split in SPECS:
        out = ROOT / "results" / ("previous_" + split)
        out.mkdir(exist_ok=True, parents=True)
        for name, value in calculate(split).items():
            (out / (name + ".json")).write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            )
