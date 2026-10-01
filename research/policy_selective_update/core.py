"""Prediction contracts and cache routing; no reference labels are consulted."""

import hashlib
import json

DECISIONS = ["supported", "contradicted", "not_established"]
ROUTES = ["recheck", "keep", "uncertain"]


def sha(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def decision(value, allowed):
    if not isinstance(value, dict) or value.get("decision") not in DECISIONS:
        return {"decision": "abstain", "evidence": [], "reason": "invalid_schema"}
    ids = value.get("evidence")
    if not isinstance(ids, list) or not ids or any(i not in allowed for i in ids):
        return {"decision": "abstain", "evidence": [], "reason": "invalid_evidence"}
    return {
        "decision": value["decision"],
        "evidence": sorted(set(ids)),
        "reason": "valid",
    }


def route(value, allowed):
    if not isinstance(value, dict) or value.get("route") not in ROUTES:
        return "uncertain"
    ids = value.get("evidence")
    if not isinstance(ids, list) or not ids or any(i not in allowed for i in ids):
        return "uncertain"
    return value["route"]


def changed_old_ids(bundle):
    current = {" ".join(s.split()) for s in bundle["after"].values()}
    return {
        i for i, s in bundle["before"].items() if " ".join(s.split()) not in current
    }


def refresh_needed(method, old, bundle, gate=None):
    if old["decision"] == "abstain" or method == "refresh_all":
        return True
    if method == "citation_text_gate":
        return bool(set(old["evidence"]) & changed_old_ids(bundle))
    if method in {"direct_scope_gate", "change_map_gate"}:
        allowed = {**bundle["before"], **bundle["after"]}
        return route(gate, allowed) != "keep"
    raise ValueError(method)


def updated(method, old, fresh, bundle, gate=None):
    refresh = refresh_needed(method, old, bundle, gate)
    return {"refreshed": refresh, "prediction": fresh if refresh else old}
