"""Connect automatically read welfare lists to the unchanged housing engine."""

import importlib.util
import json
from pathlib import Path

from reader import execute, sha

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "policy_semantic_diff"


def run(output):
    spec = importlib.util.spec_from_file_location(
        "original_semantic_engine", OLD / "engine.py"
    )
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    cases = json.loads((OLD / "cases.json").read_text())
    pred = json.loads((ROOT / "results/predictions.json").read_text())
    refined = json.loads((ROOT / "results/refined.json").read_text())
    methods = [
        "document_first",
        "tfidf_top1",
        "klue_top1",
        "tfidf_consensus3",
        "refined_posthoc",
    ]
    rows = []
    for case in cases:
        for method in methods:
            for year in ["2022", "2023"]:
                doc = "moving" + year
                rule = (
                    refined["predictions"][doc]["fields"]["welfare"]
                    if method == "refined_posthoc"
                    else pred[doc][method]["welfare"]
                )
                housing = engine.component(case["profile"], year, "housing")
                welfare = execute("welfare", rule["value"], case["profile"])
                state = engine.conjunction([housing["state"], welfare])
                reference = engine.evaluate(case["profile"], year)["state"]
                rows.append(
                    dict(
                        case=case["id"],
                        method=method,
                        year=year,
                        manual_housing=housing["state"],
                        automatic_welfare=welfare,
                        extracted_list=rule["value"],
                        evidence=rule["evidence"],
                        state=state,
                        manual_reference=reference,
                        exact=state == reference,
                    )
                )
    result = dict(
        scope="Hybrid only: housing still manually formalized; welfare lists automatically read; 14 original hypothetical cases, not new applicants.",
        original_engine_sha256=sha((OLD / "engine.py").read_bytes()),
        counts={
            m: dict(rows=28, agreed=sum(r["exact"] for r in rows if r["method"] == m))
            for m in methods
        },
        rows=rows,
    )
    Path(output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(result["counts"])


if __name__ == "__main__":
    run(ROOT / "results/bridge.json")
