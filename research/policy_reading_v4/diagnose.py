"""Descriptive error localization after observing the fixed v4 comparison."""

from collections import Counter
import json
from common import ROOT, datasets, save


def main():
    rows = datasets()["distractor"]
    roles = ["before", "after", "other"]
    output = {}
    for condition in ["raw_scope", "normalized_scope"]:
        counts = Counter()
        for seed in [17, 42, 2026]:
            predictions = json.loads(
                (ROOT / f"results/{condition}/{seed}/distractor.json").read_text()
            )
            for row, p in zip(rows, predictions):
                assert len(row["spans"]) == len(p["spans"])
                for gold, pred in zip(row["spans"], p["spans"]):
                    assert gold["start"] == pred["start"] and gold["end"] == pred["end"]
                    counts[gold["role"], pred["role"]] += 1
        output[condition] = [[counts[a, b] for b in roles] for a in roles]
    index = next(i for i, r in enumerate(rows) if r["pair_id"] == "evaluation-8-0-0-0")
    example = dict(
        row=rows[index],
        seed=42,
        predictions={
            c: json.loads((ROOT / f"results/{c}/42/distractor.json").read_text())[index]
            for c in output
        },
    )
    save(
        ROOT / "error_groups.json",
        dict(
            status="Descriptive post-result audit, no extra fitting",
            roles=roles,
            matrices=output,
            unit="One parsed amount; all three seeds pooled. Not the all-amounts exact metric.",
            fixed_example=example,
        ),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
