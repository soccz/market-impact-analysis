"""Narrow temporal-cap interpreter; query slots are supplied, not NLP-parsed."""

from datetime import date


def validate(value, allowed):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("branches"), list)
        or not value["branches"]
    ):
        return None
    result = []
    for b in value["branches"]:
        if not isinstance(b, dict):
            return None
        if type(b.get("max_krw")) is not int or b["max_krw"] < 0:
            return None
        if (
            not isinstance(b.get("evidence"), list)
            or not b["evidence"]
            or any(i not in allowed for i in b["evidence"])
        ):
            return None
        try:
            start = (
                date.fromisoformat(b["from_inclusive"])
                if b.get("from_inclusive") is not None
                else date.min
            )
            end = (
                date.fromisoformat(b["until_exclusive"])
                if b.get("until_exclusive") is not None
                else date.max
            )
        except (ValueError, TypeError):
            return None
        if start >= end:
            return None
        result.append((start, end, b["max_krw"], tuple(sorted(set(b["evidence"])))))
    result.sort()
    if result[0][0] != date.min or result[-1][1] != date.max:
        return None
    if any(a[1] != b[0] for a, b in zip(result, result[1:])):
        return None
    return result


def predict(contract, joined_on, asserted_cap):
    if contract is None:
        return "abstain"
    if joined_on is None:
        values = {b[2] for b in contract}
    else:
        try:
            day = date.fromisoformat(joined_on)
        except (ValueError, TypeError):
            return "abstain"
        values = {b[2] for b in contract if b[0] <= day < b[1]}
    if not values:
        return "abstain"
    answers = {v == asserted_cap for v in values}
    if len(answers) > 1:
        return "not_established"
    return "supported" if True in answers else "contradicted"
