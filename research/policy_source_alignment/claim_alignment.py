"""Claim-only post-hoc repair of explicit benefit coordination and scalar units.

This is a limited deterministic grammar, not an independent annotation model.
"""

import copy
import re
from bridge import fact_audit, correct_facts

BENEFITS = {
    "생계": "livelihood_benefit",
    "의료": "medical_benefit",
    "주거": "housing_benefit",
    "교육": "education_benefit",
}
NAMES = r"(?:생계|의료|주거|교육)(?:급여)?"
GROUP = re.compile(
    r"(?P<names>"
    + NAMES
    + r"(?:\s*[·ㆍ,、]\s*"
    + NAMES
    + r")*)급여\s*(?:를|는|만|도)?\s*(?:모두\s*)?(?P<verb>받지\s*않(?:는다|고)|받(?:는다|고)|수급하지\s*않는다)"
)
OWNER = re.compile(
    r"부모(?:님)?(?:은|는|의)?|배우자(?:는|의)?|친구(?:는|의)?|나는|본인은|신청자는|내\s|나의\s"
)
PERCENT = re.compile(
    r"(?:내|나의|본인(?:의)?|신청자(?:의)?)\s*(?:가구\s*)?(?:기준\s*)?중위\s*소득\s*(?:비율)?\s*(?:은|는|이|가|:)?\s*(\d+(?:\.\d+)?)\s*%"
)


def align(claim, profile):
    base_report = fact_audit(claim, profile)
    out = correct_facts(profile, base_report)
    report = dict(money=base_report, patches=[], removals=[])
    if (
        not isinstance(out, dict)
        or out.get("status") != "ready"
        or not isinstance(out.get("values"), list)
    ):
        return out, report
    values = {x["field"]: x["value"] for x in out["values"]}
    evidence = {}
    for match in GROUP.finditer(claim):
        # Resolve only an explicit applicant subject within this sentence.
        prefix = claim[: match.start()].rsplit(". ", 1)[-1].rsplit("\n", 1)[-1]
        subjects = list(OWNER.finditer(prefix))
        if not subjects or subjects[-1].group().startswith(("부모", "배우자", "친구")):
            continue
        value = "않" not in match["verb"]
        for name in re.findall(r"생계|의료|주거|교육", match["names"]):
            evidence.setdefault(BENEFITS[name], []).append(
                (value, match.start(), match.end())
            )
    for match in PERCENT.finditer(claim):
        value = float(match[1])
        evidence.setdefault("income_pct", []).append(
            (int(value) if value.is_integer() else value, match.start(), match.end())
        )
    for field, matches in evidence.items():
        if len(matches) != 1:
            continue
        value, start, end = matches[0]
        if values.get(field) != value:
            report["patches"].append(
                dict(
                    field=field,
                    before=values.get(field),
                    after=value,
                    start=start,
                    end=end,
                )
            )
            values[field] = value
    # A number explicitly expressed as KRW/age is not a residence duration.
    if (
        "parent_residence_months" in values
        and not re.search(r"거주|살았|살고|개월|년간", claim)
        and re.search(r"부모.{0,30}(?:보증금|만\s*\d+\s*세)", claim)
    ):
        report["removals"].append(
            dict(
                field="parent_residence_months",
                before=values.pop("parent_residence_months"),
                reason="explicit_non_duration_unit_without_residence_cue",
            )
        )
    out["values"] = [dict(field=k, value=v) for k, v in values.items()]
    return out, report
