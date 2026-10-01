"""Conservative, claim-only Korean monetary grounding; no policy or gold input."""

import copy, re
from fractions import Fraction

SMALL = {"천": 1000, "백": 100, "십": 10}
BIG = {"조": 10**12, "억": 10**8, "만": 10**4}
TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?|[조억만천백십]")
CUES = {
    "보증금": "deposit_krw",
    "월세": "monthly_rent_krw",
    "가구 연소득": "household_income_krw",
    "연소득": "income_krw",
}
SUBJECT = re.compile(
    r"부모(?:님)?(?:의|은|는)?|배우자(?:의|는)?|친구(?:의|는)?|(?<![가-힣])내|나의|본인(?:의|은|는)?|신청자(?:의|는)?"
)
EXTERNAL = re.compile(r"^(부모|배우자|친구)")
CUE = re.compile("|".join(map(re.escape, sorted(CUES, key=len, reverse=True))))
AMOUNT = re.compile(r"\s*(?:은|는|이|가|:)?\s*([\d,\.조억만천백십\s]+)원")
UNKNOWN = re.compile(r"\s*(?:은|는|이|가|:)?\s*(?:밝히지|모르|정보가 없|알 수 없)")


def money(raw):
    """Exact integer KRW. Unsupported/ambiguous forms raise rather than guess."""
    text = re.sub(r"\s+", "", raw).removesuffix("원")
    toks = TOKEN.findall(text)
    if not toks or "".join(toks) != text:
        raise ValueError("unsupported notation")
    total = Fraction(0)
    section = Fraction(0)
    pending = None
    last_small = 10000
    last_big = 10**16
    for t in toks:
        if t[0].isdigit():
            if pending is not None:
                raise ValueError("adjacent numbers")
            if "," in t and not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):
                raise ValueError("invalid comma")
            pending = Fraction(t.replace(",", ""))
        elif t in SMALL:
            scale = SMALL[t]
            if scale >= last_small:
                raise ValueError("small unit order")
            section += (pending if pending is not None else 1) * scale
            pending = None
            last_small = scale
        else:
            scale = BIG[t]
            if scale >= last_big:
                raise ValueError("large unit order")
            chunk = section + (pending if pending is not None else 0)
            if chunk == 0 and pending is None and section == 0:
                chunk = Fraction(1)
            total += chunk * scale
            section = Fraction(0)
            pending = None
            last_small = 10000
            last_big = scale
    value = total + section + (pending if pending is not None else 0)
    if value.denominator != 1 or not 0 <= value <= 10**15:
        raise ValueError("fractional or out of range won")
    return int(value)


def spans(claim):
    out = []
    for match in CUE.finditer(claim):
        field = CUES[match.group()]
        subjects = list(SUBJECT.finditer(claim[: match.start()]))
        subject = subjects[-1].group() if subjects else None
        # Carry explicit subject through a short coordinated phrase only.
        if subjects and any(
            x in claim[subjects[-1].end() : match.start()]
            for x in [". ", "。", ";", "\n"]
        ):
            subject = None
        owner = (
            "external"
            if subject and EXTERNAL.match(subject)
            else "applicant" if subject else "unresolved"
        )
        if field == "household_income_krw" and owner == "applicant":
            owner = "household"
        tail = claim[match.end() :]
        amount = AMOUNT.match(tail)
        missing = UNKNOWN.match(tail)
        row = {
            "field": field,
            "owner": owner,
            "cue_start": match.start(),
            "cue_end": match.end(),
        }
        if amount and re.match(
            r"\s*(?:에서|보다|부터|~|대신|또는)", tail[amount.end() :]
        ):
            row.update(status="unsupported_relation")
            out.append(row)
            continue
        if amount:
            start = match.end() + amount.start(1)
            end = match.end() + amount.end(1) + 1
            try:
                value = money(amount.group(1))
            except ValueError:
                row.update(
                    status="unsupported", start=start, end=end, text=claim[start:end]
                )
            else:
                row.update(
                    status="amount",
                    value=value,
                    start=start,
                    end=end,
                    text=claim[start:end],
                )
        elif missing:
            row.update(
                status="explicit_unknown",
                start=match.end(),
                end=match.end() + missing.end(),
                text=missing.group(),
            )
        else:
            row.update(status="unsupported")
        out.append(row)
    return out


def audit(claim, profile):
    evidence = spans(claim)
    issues = []
    patches = {}
    checked = []
    if (
        not isinstance(profile, dict)
        or profile.get("status") != "ready"
        or not isinstance(profile.get("values"), list)
    ):
        return dict(
            evidence=evidence,
            issues=[],
            patches={},
            checked=[],
            status="profile_unavailable",
        )
    vals = {}
    for x in profile["values"]:
        if (
            not isinstance(x, dict)
            or set(x) != {"field", "value"}
            or x["field"] in vals
        ):
            return dict(
                evidence=evidence,
                issues=[],
                patches={},
                checked=[],
                status="profile_invalid",
            )
        vals[x["field"]] = x["value"]
    for field in CUES.values():
        es = [e for e in evidence if e["field"] == field]
        own = [e for e in es if e["owner"] in ["applicant", "household"]]
        # Multiple mentions (even equal amounts) require discourse resolution.
        if len(own) != 1 or any(e["status"] == "unsupported" for e in own):
            continue
        e = own[0]
        if e["status"] == "amount":
            target = e["value"]
        elif e["status"] == "explicit_unknown":
            target = None
        else:
            continue
        checked.append(field)
        if vals.get(field) != target:
            issues.append(
                dict(
                    field=field,
                    predicted=vals.get(field),
                    grounded=target,
                    reason=(
                        "amount_mismatch"
                        if target is not None
                        else "invented_missing_amount"
                    ),
                    span=e,
                )
            )
            patches[field] = target
    return dict(
        evidence=evidence,
        issues=issues,
        patches=patches,
        checked=sorted(set(checked)),
        status="checked" if checked else "not_covered",
    )


def correct(profile, report):
    if not report["patches"]:
        return copy.deepcopy(profile)
    out = copy.deepcopy(profile)
    values = {x["field"]: x["value"] for x in out["values"]}
    values.update(report["patches"])
    out["values"] = [dict(field=k, value=v) for k, v in values.items()]
    return out
