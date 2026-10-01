"""Explicit caller scope contract prevents promotion to whole-program decisions.

The contract is analyst/caller input, not model output or human certification.
It is a usage boundary, not a proof that every semantic exception was detected.
"""

import re
from bridge import sha
from age_extension import coverage, compile_response, extract
from evaluate_v2 import execute
from claim_alignment import align


def read(bundle, selection, claim, profile, scope_contract=None):
    if (
        not isinstance(scope_contract, dict)
        or scope_contract.get("mode") != "bounded_clause"
    ):
        return dict(decision="abstain", reason="explicit_bounded_scope_required")
    if scope_contract.get("evidence_sha256") != sha(bundle["evidence"]):
        return dict(decision="abstain", reason="source_version_changed")
    report = coverage(bundle)
    if not report["covered"]:
        return dict(
            decision="abstain", reason="unsupported_source_grammar", coverage=report
        )
    text = "\n".join(bundle["evidence"].values())
    if re.search(r"아니다|않는다|상관없이|불문하고|제외하고는", text):
        return dict(decision="abstain", reason="unresolved_negation_or_override")
    if re.search(r"또는|하나라도|어느 하나", text) and re.search(
        r"모두|동시에|이면서|및", text
    ):
        return dict(
            decision="abstain", reason="nested_coordination_outside_one_operator"
        )
    fields = {x["atom"]["field"] for x in extract(bundle)["atoms"]}
    if not fields.issubset(set(scope_contract.get("fields", []))):
        return dict(decision="abstain", reason="field_outside_declared_scope")
    policy = compile_response(bundle, selection)
    facts, audit = align(claim, profile)
    return dict(
        decision=execute(policy, facts, list(bundle["evidence"])),
        reason="bounded_clause_only",
        scope_contract=scope_contract,
        policy=policy,
        profile=facts,
        claim_audit=audit,
    )
