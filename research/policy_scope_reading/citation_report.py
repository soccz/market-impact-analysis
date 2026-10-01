"""Separate raw semantic labels from citation-contract failures and their ablation."""

import argparse
import json
from pathlib import Path

from evaluate import parsed
from infer import ROOT
from report import enriched, export_official

METHODS = [
    m + "-" + p
    for m in ["kanana-public", "qwen"]
    for p in ["baseline", "evidence", "citation-only"]
]


def metrics(rows, records):
    predictions = {
        name: {r["id"]: parsed(values[r["id"]], r) for r in rows}
        for name, values in records.items()
    }
    result = enriched(rows, predictions)
    for name, values in records.items():
        for row in result[name]["cases"]:
            raw = values[row["id"]].get("parsed")
            row["raw_decision"] = raw.get("decision") if isinstance(raw, dict) else None
            row["raw_semantic_correct"] = row["raw_decision"] == row["reference"]
        cases = result[name]["cases"]
        result[name]["metrics"].update(
            raw_decision_correct=sum(r["raw_semantic_correct"] for r in cases),
            invalid_citation=sum(
                r["prediction"]["reason"] == "invalid_citation" for r in cases
            ),
            invalid_schema=sum(
                r["prediction"]["reason"] == "invalid_schema" for r in cases
            ),
            raw_correct_but_contract_failed=sum(
                r["raw_semantic_correct"] and r["prediction"]["decision"] == "abstain"
                for r in cases
            ),
        )
    return result


def run(cache, output=None):
    cache = Path(cache)
    out = Path(output or ROOT / "results/citation")
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for split in ["official", "replication"]:
        rows = json.loads((cache / (split + "_inputs.json")).read_text())
        records = {}
        for name in METHODS:
            folder = (
                cache / "results/citation" / split / name
                if name.endswith("citation-only")
                else ROOT / "results/extension" / split / name
            )
            records[name] = {
                r["id"]: json.loads((folder / (r["id"] + ".json")).read_text())
                for r in rows
            }
        export_official(
            {name: r for name, r in records.items() if name.endswith("citation-only")},
            out / split,
        )
        for v in ["base", "distractor", "removed"]:
            selected = [r for r in rows if r.get("variant", "base") == v]
            groups[split + "_" + v] = metrics(selected, records)
    summary = {
        g: {m: s["metrics"] for m, s in methods.items()}
        for g, methods in groups.items()
    }
    for name, data in [("summary.json", summary), ("case_scores.json", groups)]:
        (out / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                g: {
                    m: {
                        k: s[k]
                        for k in [
                            "correct",
                            "joint",
                            "raw_decision_correct",
                            "invalid_citation",
                        ]
                    }
                    for m, s in methods.items()
                }
                for g, methods in summary.items()
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
