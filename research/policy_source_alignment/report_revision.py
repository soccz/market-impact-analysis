"""Replay saved natural-language parsing, source grounding and cache repair."""

import argparse
import json
from pathlib import Path
from bridge import ROOT, MODELS, sha
from revision_chain import (
    UPDATE,
    validate,
    ground_contract,
    claim_slots,
    update,
    answer,
)
from revision_run import evidence, jobs
from report_transfer import public_record


def calculate(records, source, contracts):
    cases = json.loads((ROOT / "revision_reference.json").read_text())["cases"]
    lookup = {(r["model"], r["stage"], r["id"]): r["parsed"] for r in records}
    rows = []
    audit = []
    rotate = {
        "supported": "contradicted",
        "contradicted": "not_established",
        "not_established": "supported",
        "abstain": "supported",
    }
    for model in MODELS:
        raw = {
            phase: validate(
                contracts[model]["contract"][phase]["parsed"], list(source[phase])
            )
            for phase in ["before", "after"]
        }
        grounded = {}
        for phase in ["before", "after"]:
            grounded[phase], a = ground_contract(
                source[phase], contracts[model]["contract"][phase]["parsed"]
            )
            audit.append(dict(model=model, phase=phase, **a))
        for i, c in enumerate(cases):
            predicted = lookup[model, "slots", c["id"]]
            parsed = claim_slots(c["claim"])
            direct_before = lookup[model, "before", c["id"]]
            direct_after = lookup[model, "after", c["id"]]
            before = (
                direct_before.get("decision", "abstain")
                if isinstance(direct_before, dict)
                else "abstain"
            )
            after = (
                direct_after.get("decision", "abstain")
                if isinstance(direct_after, dict)
                else "abstain"
            )
            for condition in ["saved", "corrupted"]:
                stored = (
                    rotate[before]
                    if condition == "corrupted" and i % 3 == 0
                    else before
                )
                result = update(grounded["before"], grounded["after"], parsed, stored)
                rows.append(
                    dict(
                        id=c["id"],
                        model=model,
                        condition=condition,
                        claim=c["claim"],
                        before_label=c["before"],
                        label=c["after"],
                        reference_slots=c["reference_slots"],
                        model_slots=predicted,
                        grounded_slots=parsed,
                        slots_exact=predicted == c["reference_slots"],
                        grounded_exact=parsed == c["reference_slots"],
                        direct_before=before,
                        direct_after=after,
                        raw_contract_model_slots=answer(raw["after"], predicted),
                        grounded_contract_model_slots=answer(
                            grounded["after"], predicted
                        ),
                        **result
                    )
                )
    summaries = {}
    for model in MODELS:
        summaries[model] = {}
        for condition in ["saved", "corrupted"]:
            rr = [
                r for r in rows if r["model"] == model and r["condition"] == condition
            ]
            summaries[model][condition] = dict(
                n=len(rr),
                before_correct=sum(r["direct_before"] == r["before_label"] for r in rr),
                stored_after_correct=sum(r["stored"] == r["label"] for r in rr),
                direct_after_correct=sum(r["direct_after"] == r["label"] for r in rr),
                raw_pipeline_correct=sum(
                    r["raw_contract_model_slots"] == r["label"] for r in rr
                ),
                grounded_contract_correct=sum(
                    r["grounded_contract_model_slots"] == r["label"] for r in rr
                ),
                updated_correct=sum(r["updated"] == r["label"] for r in rr),
                model_slots_exact=sum(r["slots_exact"] for r in rr),
                grounded_slots_exact=sum(r["grounded_exact"] for r in rr),
                review=sum(r["review"] for r in rr),
                decision_changes=sum(r["revision_changes_decision"] for r in rr),
                prior_errors=sum(r["prior_answer_disagrees"] for r in rr),
                stable_prior_errors=sum(
                    r["prior_answer_disagrees"] and not r["revision_changes_decision"]
                    for r in rr
                ),
                fixes=sum(
                    r["stored"] != r["label"] and r["updated"] == r["label"] for r in rr
                ),
                harms=sum(
                    r["stored"] == r["label"] and r["updated"] != r["label"] for r in rr
                ),
            )
    return dict(rows=rows, summary=summaries, contract_audit=audit)


def export(previous_cache, cache, dest):
    source = evidence(previous_cache)
    records = []
    for model in MODELS:
        for stage, key, req in jobs(source, model):
            raw = json.loads(
                (Path(cache) / model / stage / (key + ".json")).read_text()
            )
            assert raw["request_sha256"] == sha(req) and raw["input_sha256"] == sha(
                req["messages"]
            )
            records.append(public_record(raw))
    contracts = json.loads((UPDATE / "results/contract/predictions.json").read_text())
    result = dict(predictions=records, **calculate(records, source, contracts))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for k, v in result.items():
        (dest / (k + ".json")).write_text(
            json.dumps(v, ensure_ascii=False, indent=2) + "\n"
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--previous-cache", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--dest", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            export(a.previous_cache, a.cache, a.dest)["summary"], ensure_ascii=False
        )
    )
