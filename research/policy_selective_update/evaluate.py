"""Distinguish correct updates, needless flips, missed changes and stale cache errors."""

from core import decision, route
from routing import GATES, METHODS, updated


def score(data, records):
    bundles = {b["id"]: b for b in data["bundles"]}
    result = {}
    for method in METHODS:
        cases = []
        for case in data["cases"]:
            b = bundles[case["bundle"]]
            r = case["reference"]
            key = case["id"]
            allowed = {**b["before"], **b["after"]}
            old = decision(records["before"][key]["parsed"], b["before"])
            fresh = decision(records["after"][key]["parsed"], allowed)
            gate = records[GATES[method]][key]["parsed"] if method in GATES else None
            current = updated(method, old, fresh, b, gate, key)
            pred = current["prediction"]
            refreshed = current["refreshed"]
            old_ok = old["decision"] == r["before"]
            now_ok = pred["decision"] == r["after"]
            stable = r["before"] == r["after"]
            predicted_flip = old["decision"] != pred["decision"] and "abstain" not in {
                old["decision"],
                pred["decision"],
            }
            cases.append(
                {
                    "id": key,
                    "bundle": b["id"],
                    "reference": r,
                    "old": old,
                    "fresh": fresh,
                    "final": pred,
                    "gate": route(gate, allowed) if method in GATES else None,
                    "refreshed": refreshed,
                    "old_correct": old_ok,
                    "correct": now_ok,
                    "pair_correct": old_ok and now_ok,
                    "reference_changed": not stable,
                    "predicted_flip": predicted_flip,
                    "false_flip": stable and predicted_flip,
                    "missed_flip": not stable and not predicted_flip,
                    "required_recheck": r["route"] != "keep",
                    "missed_recheck": r["route"] != "keep" and not refreshed,
                    "unnecessary_recheck": r["route"] == "keep" and refreshed,
                    "stale_error": not stable
                    and old_ok
                    and not refreshed
                    and not now_ok,
                    "retained_old_error": not old_ok and not refreshed and not now_ok,
                    "corrected_old_error": not old_ok and refreshed and now_ok,
                    "harmful_refresh": stable and old_ok and refreshed and not now_ok,
                    "fresh_joint": fresh["decision"] == r["after"]
                    and set(r["after_evidence"]).issubset(fresh["evidence"]),
                    "old_joint": old_ok
                    and set(r["before_evidence"]).issubset(old["evidence"]),
                }
            )
        totals = {
            k: sum(row[k] for row in cases)
            for k in [
                "old_correct",
                "correct",
                "pair_correct",
                "refreshed",
                "reference_changed",
                "predicted_flip",
                "false_flip",
                "missed_flip",
                "required_recheck",
                "missed_recheck",
                "unnecessary_recheck",
                "stale_error",
                "retained_old_error",
                "corrected_old_error",
                "harmful_refresh",
                "fresh_joint",
                "old_joint",
            ]
        }
        totals.update(
            n=len(cases),
            reference_stable=sum(not r["reference_changed"] for r in cases),
            keep_eligible=sum(not r["required_recheck"] for r in cases),
            abstain=sum(r["final"]["decision"] == "abstain" for r in cases),
        )
        totals["required_recheck_recall"] = (
            (totals["required_recheck"] - totals["missed_recheck"])
            / totals["required_recheck"]
            if totals["required_recheck"]
            else None
        )
        result[method] = {"metrics": totals, "cases": cases}
    return result
