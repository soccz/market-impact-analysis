"""A small rerun check; agreement on two failed outputs is not a correct answer."""

import argparse
import json
from pathlib import Path

from evaluate import parsed
from infer import ROOT, digest
from report import export_official


def run(cache, output=None):
    cache = Path(cache)
    out = Path(output or ROOT / "results/repeat")
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((cache / "repeat_inputs.json").read_text())
    protocol = json.loads((ROOT / "repeat_protocol.json").read_text())
    assert [r["id"] for r in rows] == protocol["case_ids"]
    raw = {}
    summary = {}
    details = {}
    for name, first_folder in [
        ("qwen-thinking", cache / "results/thinking/replication/qwen-thinking"),
        (
            "kanana-public-baseline",
            cache / "results/extension/replication/kanana-public-baseline",
        ),
    ]:
        raw[name] = {}
        details[name] = []
        for row in rows:
            first = json.loads((first_folder / (row["id"] + ".json")).read_text())
            again = json.loads(
                (cache / "results/repeat" / name / (row["id"] + ".json")).read_text()
            )
            assert (
                first["request_sha256"] == again["request_sha256"]
                and first["model_digest"] == again["model_digest"]
            )
            a, b = parsed(first, row), parsed(again, row)
            raw[name][row["id"]] = again
            details[name].append(
                dict(
                    id=row["id"],
                    first=a,
                    repeat=b,
                    first_valid=a["decision"] != "abstain",
                    repeat_valid=b["decision"] != "abstain",
                    same_decision=a["decision"] == b["decision"],
                    same_decision_and_evidence=a == b,
                    both_abstain=a["decision"] == b["decision"] == "abstain",
                    same_final_text=digest(
                        first["response"]["message"].get("content", "").encode()
                    )
                    == digest(again["response"]["message"].get("content", "").encode()),
                )
            )
        summary[name] = {
            k: sum(r[k] for r in details[name])
            for k in [
                "first_valid",
                "repeat_valid",
                "same_decision",
                "same_decision_and_evidence",
                "both_abstain",
                "same_final_text",
            ]
        }
        summary[name]["n"] = len(rows)
    export_official(raw, out)
    for name, data in [("summary.json", summary), ("cases.json", details)]:
        (out / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    run(a.cache, a.output)
