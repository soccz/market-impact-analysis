"""Post-hoc audit of answer correctness masking applicant-fact errors."""

import json
from fractions import Fraction
from pathlib import Path
from diagnostics import SPLITS, dataset

P = Path(__file__).parent


def calculate():
    result = {}
    for split in SPLITS:
        data = dataset(split)
        cases = {c["id"]: c for c in data["cases"]}
        preds = json.loads((P / f"results/{split}/predictions.json").read_text())
        profiles = {
            (r["model"], r["id"]): r["parsed"] for r in preds if r["stage"] == "profile"
        }
        rows = json.loads((P / f"results/{split}/rows.json").read_text())
        result[split] = {}
        for model in ["qwen", "kanana-public"]:
            groups = {}
            for bundle in sorted({r["bundle"] for r in rows}):
                rr = [r for r in rows if r["model"] == model and r["bundle"] == bundle]
                errors = []
                for r in rr:
                    if r["profile_exact"]:
                        continue
                    p = profiles[model, r["id"]]
                    q = cases[r["id"]]["reference_profile"]
                    pv = (
                        {
                            v["field"]: v["value"]
                            for v in p.get("values", [])
                            if v.get("value") is not None
                        }
                        if isinstance(p, dict)
                        else {}
                    )
                    qv = {
                        v["field"]: v["value"]
                        for v in q["values"]
                        if v["value"] is not None
                    }
                    differences = [
                        dict(field=f, predicted=pv.get(f), reference=qv.get(f))
                        for f in sorted(set(pv) | set(qv))
                        if pv.get(f) != qv.get(f)
                    ]
                    errors.append(
                        dict(
                            id=r["id"],
                            answer_correct=r["pipeline"]["decision"] == r["label"],
                            differences=differences,
                            predicted_assertion=(
                                p.get("assertion") if isinstance(p, dict) else None
                            ),
                            reference_assertion=q["assertion"],
                            profile_status=r["pipeline"].get("profile_status"),
                        )
                    )
                groups[bundle] = dict(
                    n=len(rr),
                    answer_correct=sum(
                        r["pipeline"]["decision"] == r["label"] for r in rr
                    ),
                    profile_exact=sum(r["profile_exact"] for r in rr),
                    joint_answer_and_profile=sum(
                        r["pipeline"]["decision"] == r["label"] and r["profile_exact"]
                        for r in rr
                    ),
                    correct_answer_wrong_profile=sum(
                        r["pipeline"]["decision"] == r["label"]
                        and not r["profile_exact"]
                        for r in rr
                    ),
                    errors=errors,
                )
            result[split][model] = groups
    # Illustrative exact arithmetic under the saved Qwen formula, not another LLM call.
    saved = json.loads((P / "results/capability_extended/predictions.json").read_text())
    policy = next(
        r["parsed"]
        for r in saved
        if r["model"] == "qwen" and r["stage"] == "policy" and r["id"] == "cap-rent"
    )
    formula = policy["derived"][0]

    def converted(deposit):
        values = {"monthly_rent_krw": 700000, "deposit_krw": deposit}
        return str(
            Fraction(str(formula["constant"]))
            + sum(
                Fraction(values[t["field"]])
                * Fraction(str(t["numerator"]))
                / Fraction(str(t["denominator"]))
                for t in formula["terms"]
            )
        )

    result["boundary_example"] = dict(
        id="cap-rent-4",
        monthly_rent_krw=700000,
        reference_deposit_krw=24000240,
        predicted_deposit_krw=24002400,
        reference_converted_krw=converted(24000240),
        predicted_converted_krw=converted(24002400),
        interpretation="Both exceed 800000, so the wrong parsed amount still yields the reference decision. This diagnostic was added after inspecting the UI trace.",
    )
    return result


if __name__ == "__main__":
    d = calculate()
    (P / "results/fact_audit.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(d["capability_extended"]["qwen"]["cap-rent"], ensure_ascii=False))
