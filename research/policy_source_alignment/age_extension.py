"""Post-transfer repair: bare age bounds and explicit inclusive age ranges.

The first Changwon/Suwon run is preserved separately. Only a manually selected
age scope enables this extension. It cannot bind ages to multiple people.
"""

import copy
import re
from candidates import extract as prior_extract, compile_selection, OPS
from source_method import request as prior_request
from explicit_logic import as_selection
from bridge import obj, arr, enum

BOUND = re.compile(r"(?<!\d)(\d{1,3})\s*세\s*(이하|미만|이상|초과)")
RANGE = re.compile(r"(?<!\d)(\d{1,3})\s*세?\s*[~∼〜–-]\s*(\d{1,3})\s*세\s*(미만|이하)?")


def extract(bundle):
    catalog = prior_extract(bundle)
    if not re.search(r"연령|나이", bundle["scope"]):
        return catalog
    for key, text in bundle["evidence"].items():
        if re.search(r"부모|배우자|자녀|부부", text):
            catalog["warnings"].append(
                dict(source=key, reason="age_subject_binding_unsupported")
            )
            continue
        covered = []

        def add(value, op, start, end):
            atom = dict(field="age_years", op=op, value=value, evidence=[key])
            if any(a["atom"] == atom for a in catalog["atoms"]):
                return
            catalog["atoms"].append(
                dict(
                    id="a" + str(len(catalog["atoms"])),
                    atom=atom,
                    source=key,
                    start=start,
                    end=end,
                    text=text[start:end],
                    kind="age_bound",
                    origin="source",
                )
            )
            covered.append((start, end))

        for m in BOUND.finditer(text):
            add(int(m[1]), OPS[m[2]], m.start(), m.end())
        for m in RANGE.finditer(text):
            if int(m[1]) > int(m[2]):
                catalog["warnings"].append(
                    dict(
                        source=key,
                        start=m.start(),
                        end=m.end(),
                        reason="reversed_age_range",
                    )
                )
                continue
            add(int(m[1]), "ge", m.start(), m.end())
            add(int(m[2]), "lt" if m[3] == "미만" else "le", m.start(), m.end())
        catalog["warnings"] = [
            w
            for w in catalog["warnings"]
            if not (
                w.get("source") == key
                and w.get("reason") == "numeric_field_unresolved"
                and any(a <= w["start"] and w["end"] <= b for a, b in covered)
            )
        ]
    return catalog


def coverage(bundle):
    try:
        catalog = extract(bundle)
    except (ValueError, ZeroDivisionError, OverflowError):
        return dict(covered=False, reasons=["invalid_numeric_expression"])
    reasons = []
    if not catalog["atoms"]:
        reasons.append("no_supported_atom")
    if catalog["warnings"]:
        reasons.append("unresolved_candidate")
    if any(re.search(r"다만|예외|제외하되", t) for t in bundle["evidence"].values()):
        reasons.append("exception_requires_nested_logic")
    return dict(covered=not reasons, reasons=reasons)


def compile_response(bundle, parsed):
    if not coverage(bundle)["covered"]:
        return dict(status="insufficient", clauses=[], derived=[])
    return compile_selection(as_selection(parsed), extract(bundle))


def request(bundle, case, model, stage):
    req = prior_request(bundle, case, model, stage)
    if stage != "policy":
        return req
    import json

    payload = json.loads(req["messages"][1]["content"])
    catalog = extract(bundle)
    payload["candidates"] = catalog
    req["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    req["format"]["properties"]["conditions"] = arr(
        enum([x["id"] for x in catalog["atoms"]] or ["__none__"])
    )
    return req
