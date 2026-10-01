"""Conservative source-literal candidates; no claims or reference labels as input."""

import re, copy
from fractions import Fraction
from bridge import money

ALIASES = {
    "deposit_krw": r"(?:임차)?보증금",
    "monthly_rent_krw": r"월\s*(?:세|임차료|임대료)",
    "income_pct": r"(?:기준\s*)?중위\s*소득",
    "household_income_krw": r"(?:가구|부부|부부합산)\s*(?:합산\s*)?연\s*소득",
    "income_krw": r"연\s*소득",
    "residence_months": r"(?:연속\s*)?거주(?:기간)?",
    "age_years": r"(?:만\s*나이|만|연령|나이)",
}
BOUND = re.compile(
    r"(?P<value>\d[\d,.조억만천백십\s]*?)\s*(?P<unit>원|%|세|개월|년)\s*(?P<op>이하|미만|이상|초과)"
)
OPS = {"이하": "le", "미만": "lt", "이상": "ge", "초과": "gt"}
BENEFITS = {
    "생계": "livelihood_benefit",
    "의료": "medical_benefit",
    "주거": "housing_benefit",
    "교육": "education_benefit",
}


def extract(bundle):
    evidence = bundle["evidence"]
    scope = bundle["scope"]
    atoms = []
    derived = []
    warnings = []

    def add(field, op, value, key, start, end, kind, origin="source"):
        atom = dict(field=field, op=op, value=value, evidence=[key])
        if any(a["atom"] == atom for a in atoms):
            return
        atoms.append(
            dict(
                id="a" + str(len(atoms)),
                atom=atom,
                source=key,
                start=start,
                end=end,
                text=evidence[key][start:end],
                kind=kind,
                origin=origin,
            )
        )

    for key, text in evidence.items():
        for match in BOUND.finditer(text):
            raw, unit = match["value"], match["unit"]
            prefix = text[max(0, match.start() - 80) : match.start()]
            prefix = prefix.rsplit("\n", 1)[-1]
            candidates = []
            allowed = (
                [
                    "deposit_krw",
                    "monthly_rent_krw",
                    "household_income_krw",
                    "income_krw",
                ]
                if unit == "원"
                else (
                    ["income_pct"]
                    if unit == "%"
                    else ["age_years"] if unit == "세" else ["residence_months"]
                )
            )
            for field in allowed:
                for alias in re.finditer(ALIASES[field], prefix):
                    candidates.append((alias.end(), alias.start(), field))
            if (
                not candidates
                and unit in ["개월", "년"]
                and re.search(
                    ALIASES["residence_months"], text[match.end() : match.end() + 24]
                )
            ):
                candidates.append((0, 0, "residence_months"))
            if not candidates:
                warnings.append(
                    dict(
                        source=key,
                        start=match.start(),
                        end=match.end(),
                        reason="numeric_field_unresolved",
                    )
                )
                continue
            _, _, field = max(candidates, key=lambda x: (x[0], -x[1]))
            try:
                value = (
                    money(raw + "원")
                    if unit == "원"
                    else Fraction(raw.replace(",", "").strip())
                    * (12 if unit == "년" else 1)
                )
                value = int(value) if Fraction(value).denominator == 1 else float(value)
            except (ValueError, ZeroDivisionError):
                warnings.append(
                    dict(
                        source=key,
                        start=match.start(),
                        end=match.end(),
                        reason="unsupported_number",
                    )
                )
                continue
            add(
                field, OPS[match["op"]], value, key, match.start(), match.end(), "bound"
            )
        absent = re.search(r"월세\s*가?\s*없는[^\n,.]{0,35}(?:불가|제외)", text)
        if absent:
            add(
                "monthly_rent_krw",
                "gt",
                0,
                key,
                absent.start(),
                absent.end(),
                "absence_exclusion",
            )
        if re.search(r"(?:기초생활|급여|수급)", text):
            excluded = bool(re.search(r"제외|불가|받지\s*않", text))
            scope_excluded = not excluded and bool(
                re.search(r"(?:급여|수급)[^。\.]{0,35}제외", scope)
            )
            if excluded or scope_excluded:
                for word, field in BENEFITS.items():
                    found = re.search(word + r"(?:급여)?", text)
                    if found:
                        add(
                            field,
                            "eq",
                            False,
                            key,
                            found.start(),
                            found.end(),
                            "benefit_exclusion",
                            "scope" if scope_excluded else "source",
                        )
        # A deliberately narrow, explicit formula grammar. Percent is normalized exactly.
        formula = re.search(
            r"보증금\s*\+\s*\[?\s*\(?\s*월\s*(?:임대료|임차료|세)\s*[×x*]\s*(\d+(?:\.\d+)?)\s*\)?\s*[÷/]\s*(\d+(?:\.\d+)?)\s*%",
            text,
        )
        if formula:
            coefficient = Fraction(formula[1]) * 100 / Fraction(formula[2])
            definition = dict(
                field="derived_0",
                constant=0,
                terms=[
                    dict(field="deposit_krw", numerator=1, denominator=1),
                    dict(
                        field="monthly_rent_krw",
                        numerator=coefficient.numerator,
                        denominator=coefficient.denominator,
                    ),
                ],
                evidence=[key],
            )
            derived.append(
                dict(
                    definition=definition,
                    source=key,
                    start=formula.start(),
                    end=formula.end(),
                    text=text[formula.start() : formula.end()],
                )
            )
            upper = [
                a
                for a in atoms
                if a["atom"]["field"] == "deposit_krw"
                and a["atom"]["op"] in ["le", "lt"]
            ]
            for bound in upper:
                add(
                    "derived_0",
                    bound["atom"]["op"],
                    bound["atom"]["value"],
                    key,
                    formula.start(),
                    formula.end(),
                    "converted_bound",
                )
    # Metadata is part of the method's coverage, never evidence of source correctness.
    return dict(atoms=atoms, derived=derived, warnings=warnings)


def compile_selection(selection, catalog):
    empty = dict(status="insufficient", clauses=[], derived=[])
    if not isinstance(selection, dict) or selection.get("status") != "ready":
        return empty
    clauses = selection.get("clauses")
    lookup = {a["id"]: a["atom"] for a in catalog["atoms"]}
    if not isinstance(clauses, list) or not 1 <= len(clauses) <= 12:
        return empty
    output = []
    for clause in clauses:
        if (
            not isinstance(clause, list)
            or not 1 <= len(clause) <= 16
            or any(type(x) is not str or x not in lookup for x in clause)
        ):
            return empty
        output.append([copy.deepcopy(lookup[x]) for x in dict.fromkeys(clause)])
    used = {a["field"] for c in output for a in c}
    return dict(
        status="ready",
        clauses=output,
        derived=[
            copy.deepcopy(d["definition"])
            for d in catalog["derived"]
            if d["definition"]["field"] in used
        ],
    )


def rule_selection(bundle, catalog):
    """Restricted grammar baseline; default conjunction is an explicit assumption."""
    text = "\n".join(bundle["evidence"].values())
    if not catalog["atoms"] or re.search(r"다만|예외|제외하되", text):
        return dict(status="insufficient", clauses=[])
    converted = any(a["kind"] == "converted_bound" for a in catalog["atoms"])
    normal = [
        a
        for a in catalog["atoms"]
        if a["kind"] != "benefit_exclusion"
        and not (converted and a["atom"]["field"] == "deposit_krw")
    ]
    excluded = [a["id"] for a in catalog["atoms"] if a["kind"] == "benefit_exclusion"]
    use_or = bool(re.search(r"하나라도|어느\s*하나|중\s*하나", text))
    groups = (
        [[a["id"]] + excluded for a in normal]
        if use_or
        else [[a["id"] for a in normal] + excluded]
    )
    return dict(status="ready", clauses=groups)


def public_catalog(catalog):
    from bridge import sha

    result = copy.deepcopy(catalog)
    for row in result["atoms"] + result["derived"]:
        row["text_sha256"] = sha(row.pop("text").encode())
    return result
