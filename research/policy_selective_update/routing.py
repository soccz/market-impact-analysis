"""Development-inspired cache audit, fixed before new-document evaluation."""

import hashlib

from core import updated as original_update

METHODS = [
    "refresh_all",
    "citation_text_gate",
    "hash_audit",
    "direct_scope_gate",
    "change_map_gate",
    "audited_change_map",
]
GATES = {
    "direct_scope_gate": "direct_gate",
    "change_map_gate": "map_gate",
    "audited_change_map": "map_gate",
}


def audit_selected(case_id):
    # Fixed one-in-four hash partition. This is not stratified on labels or errors.
    return (
        int(
            hashlib.sha256(
                ("policy-update-audit-20261001:" + case_id).encode()
            ).hexdigest(),
            16,
        )
        % 4
        == 0
    )


def updated(method, old, fresh, bundle, gate, case_id):
    if method == "hash_audit":
        refresh = old["decision"] == "abstain" or audit_selected(case_id)
        reason = "audit_or_invalid_cache" if refresh else "hash_keep"
    elif method == "audited_change_map":
        base = original_update("change_map_gate", old, fresh, bundle, gate)
        refresh = (
            base["refreshed"]
            or old["decision"] == "not_established"
            or audit_selected(case_id)
        )
        reason = "map_gate_or_unknown_cache_or_audit" if refresh else "map_keep"
    else:
        result = original_update(method, old, fresh, bundle, gate)
        result["routing_reason"] = method
        return result
    return {
        "refreshed": refresh,
        "prediction": fresh if refresh else old,
        "routing_reason": reason,
    }
