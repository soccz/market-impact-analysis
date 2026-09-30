"""Compose two known perturbations, without changing the frozen v3 input files."""

from copy import deepcopy
from common import ROOT, V3, read_rows, save
from representation import money_view, transform_row
import hashlib
import json


def main():
    assert not (ROOT / "freeze.json").exists(), "Do not alter frozen inputs"
    rows = []
    for row in read_rows(V3 / "data/distractor.jsonl"):
        view = money_view(row["text"], lambda s: f"{int(s['won']):,}원")
        new = transform_row(row, view)
        new["condition"] = "combined"
        new["id"] = hashlib.sha256((row["pair_id"] + "-combined").encode()).hexdigest()[
            :16
        ]
        rows.append(new)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/combined.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    # No relation gold is assigned: this demonstrates information hidden from the encoder.
    loss = []
    for value in ["500,000원", "600,000원"]:
        text = f"가람지원금의 기준 안내는 50만 원, 후속 안내는 {value}이다."
        view = money_view(text)
        loss.append(
            dict(
                text=text,
                model_text=view["text"],
                values_won=[s["won"] for s in view["amounts"]],
            )
        )
    assert loss[0]["model_text"] == loss[1]["model_text"]
    assert loss[0]["values_won"] != loss[1]["values_won"]
    save(
        ROOT / "data/value_information.json",
        dict(
            cases=loss,
            interpretation="Identical encoder text can have different amounts. Values remain in a side record and numeric decoder, not in encoder features. This is a tradeoff demonstration, not an accuracy benchmark.",
        ),
    )
    print("Prepared 576 combined probes and a value-information tradeoff example")


if __name__ == "__main__":
    main()
