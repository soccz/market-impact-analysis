"""Frozen transfer method: explicit operator with a conservative coverage gate."""

import copy
import re
from bridge import infer, MODELS
from candidates import extract, compile_selection
from explicit_logic import request as explicit_request, as_selection


def coverage(bundle):
    try:
        catalog = extract(bundle)
    except (ValueError, ZeroDivisionError, OverflowError):
        return dict(covered=False, reasons=["invalid_numeric_expression"])
    reasons = []
    if not catalog["atoms"]:
        reasons.append("no_supported_atom")
    if catalog["warnings"]:
        reasons.append("unresolved_numeric_field")
    if any(
        re.search(r"다만|예외|제외하되", text) for text in bundle["evidence"].values()
    ):
        reasons.append("exception_requires_nested_logic")
    return dict(covered=not reasons, reasons=reasons)


def compile_response(bundle, parsed):
    report = coverage(bundle)
    if not report["covered"]:
        return dict(status="insufficient", clauses=[], derived=[])
    return compile_selection(as_selection(parsed), extract(bundle))


def request(bundle, case, model, stage):
    if stage == "policy":
        return explicit_request(bundle, case, model, stage)
    # Frozen prior claim parser and direct decision control. Source is never
    # added to profile requests; gold labels are never included in any request.
    from bridge import old_request

    return old_request(bundle, case, model, stage)


def run(inputs, out, model):
    original = infer.request
    try:
        infer.request = request
        infer.run(inputs, out, model)
    finally:
        infer.request = original


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    for model in MODELS:
        run(a.inputs, str(Path(a.cache) / model), model)
