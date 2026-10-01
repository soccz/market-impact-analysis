"""Reject internally inconsistent generated traces without reading gold labels.

This does not prove that a trace is faithful to its source. It is a limited
selective check, learned from development failures and frozen before test-output
inspection. No generated comparison is used to replace a semantic answer.
"""

import operator
import re

OPS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "≤": operator.le,
    "≥": operator.ge,
    "=": operator.eq,
}


def apply(case, record, prediction):
    trace = (record.get("parsed") or {}).get("rule", "")
    reasons = []
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(<=|>=|≤|≥|<|>|=)\s*(\d+(?:\.\d+)?)", trace
    ):
        if not OPS[m[2]](float(m[1]), float(m[3])):
            reasons.append("false_generated_inequality")
    positive = (
        re.search(r"충족한다[.。]?$", "".join(case["claim"].split()))
        and "충족하지" not in case["claim"]
    )
    if (
        positive
        and prediction["decision"] == "supported"
        and any(word in trace for word in ["불충족", "미충족"])
    ):
        reasons.append("positive_claim_but_trace_says_failure")
    if reasons:
        return dict(
            decision="abstain", evidence=[], reason="+".join(sorted(set(reasons)))
        )
    return dict(prediction)
