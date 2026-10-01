"""A bounded post-hoc check of an epistemic versus meta-claim label convention."""

import json
from pathlib import Path

from infer import ROOT

CASES = {
    "development": ["dev-possibility-1"],
    "official_base": ["jeju-possible-deadline"],
    "official_distractor": ["jeju-possible-deadline-distractor"],
}


def run(output=None):
    result = {
        "status": "Post-hoc annotation-convention sensitivity; not corrected gold and not independent human adjudication.",
        "question": "The primary convention labels an uncertain policy outcome not_established. A meta-claim that the available evidence permits certainty can instead be read as contradicted by an explicit possibility clause. Only the two named wording families are varied; all original labels and outputs remain unchanged.",
        "alternative_decision": "contradicted",
        "groups": {},
    }
    for stage in ["", "extension"]:
        groups = json.loads((ROOT / "results" / stage / "case_scores.json").read_text())
        target = {}
        for group, ids in CASES.items():
            target[group] = {}
            for method, s in groups[group].items():
                rows = s["cases"]
                selected = [r for r in rows if r["id"] in ids]
                assert len(selected) == len(ids) and all(
                    r["reference"] == "not_established" for r in selected
                )
                delta = sum(
                    int(r["prediction"]["decision"] == "contradicted")
                    - int(r["correct"])
                    for r in selected
                )
                target[group][method] = dict(
                    total=len(rows),
                    original_correct=s["metrics"]["correct"],
                    alternative_correct=s["metrics"]["correct"] + delta,
                    changed_label_ids=ids,
                )
        result["groups"][stage or "first"] = target
    Path(output or ROOT / "results/annotation_sensitivity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    run()
