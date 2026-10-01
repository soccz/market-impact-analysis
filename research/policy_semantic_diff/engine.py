"""Execute explicitly authored, partial policy rules. No NLP extraction or award decision."""

from itertools import combinations

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"
BENEFITS = {"education", "livelihood", "medical", "housing"}


def conjunction(values):
    return FAIL if FAIL in values else UNKNOWN if UNKNOWN in values else PASS


def consensus(values):
    return values[0] if len(set(values)) == 1 else UNKNOWN


def validate(profile):
    for name in ("deposit_won", "rent_won"):
        v = profile.get(name)
        if v is not None and (type(v) is not int or v < 0):
            raise ValueError(f"{name}: use a nonnegative integer in KRW or null")
    b = profile.get("benefits")
    if b is not None and (
        not isinstance(b, list)
        or any(x not in BENEFITS for x in b)
        or len(set(b)) != len(b)
    ):
        raise ValueError("benefits: unique supported categories, [] or null")


def housing_2022(deposit, rent, interpretation):
    # Both readings retain the explicit independent deposit exclusion.
    if deposit is not None and deposit > 50_000_000:
        return FAIL
    if rent is not None and rent > 550_000:
        return FAIL
    if deposit is None or rent is None:
        return UNKNOWN
    if rent <= 400_000:
        return PASS
    if interpretation == "thousand_floor":
        # Candidate reading of '천원 단위 절사', not an official clarification.
        converted = (deposit * 375 // 120_000_000) * 1000
        return PASS if converted + rent <= 550_000 else FAIL
    if interpretation == "published_table":
        # Explicit 15 bands in notice 2022-2542, section 2: [0, 3.2m), ...
        band = deposit // 3_200_000
        return PASS if band < 15 and rent <= 550_000 - band * 10_000 else FAIL
    raise ValueError(interpretation)


def housing_2023(deposit, rent):
    if deposit is not None and deposit > 200_000_000:
        return FAIL, None
    if rent is not None and rent > 2_000_000:
        return FAIL, None
    if deposit is None or rent is None:
        return UNKNOWN, None
    standard = deposit + rent * 100
    transaction = deposit + rent * 70 if standard < 50_000_000 else standard
    return (PASS if transaction <= 200_000_000 else FAIL), transaction


def component(profile, version, kind, hidden=(), ablation=None):
    evidence = f"{version}-{kind}"
    result = {"evidence": [evidence], "state": UNKNOWN, "reason": "evidence_hidden"}
    if evidence in hidden:
        return result
    if kind == "welfare":
        benefits = profile.get("benefits")
        excluded = (
            {"housing"}
            if version == "2022" or ablation == "keep_old_welfare"
            else {"housing", "livelihood", "medical"}
        )
        result.update(excluded=sorted(excluded))
        if benefits is None:
            result["reason"] = "missing_benefits"
        else:
            result.update(
                state=FAIL if excluded.intersection(benefits) else PASS,
                reason="exclusion_list",
            )
        return result
    deposit, rent = profile.get("deposit_won"), profile.get("rent_won")
    if version == "2022":
        if ablation == "drop_exception":
            values = [
                (
                    FAIL
                    if deposit is not None
                    and deposit > 50_000_000
                    or rent is not None
                    and rent > 400_000
                    else UNKNOWN if deposit is None or rent is None else PASS
                )
            ]
            result.update(state=values[0], reason="exception_removed")
        else:
            readings = {
                name: housing_2022(deposit, rent, name)
                for name in ("thousand_floor", "published_table")
            }
            state = consensus(list(readings.values()))
            reason = (
                "interpretation_disagreement"
                if len(set(readings.values())) > 1
                else "missing_housing" if state == UNKNOWN else "housing_rule"
            )
            result.update(state=state, reason=reason, readings=readings)
    else:
        state, transaction = housing_2023(deposit, rent)
        result.update(
            state=state,
            reason="missing_housing" if state == UNKNOWN else "transaction_rule",
            transaction_won=transaction,
        )
    return result


def evaluate(profile, version, hidden=(), ablation=None, versions=None):
    validate(profile)
    if version not in ("2022", "2023"):
        raise ValueError(version)
    versions = versions or {"housing": version, "welfare": version}
    traces = {
        kind: component(profile, versions[kind], kind, hidden, ablation)
        for kind in ("housing", "welfare")
    }
    state = conjunction([x["state"] for x in traces.values()])
    # Unknown evidence is not a failed condition. A definite failed AND term is decisive.
    questions = (
        [x["reason"] for x in traces.values() if x["state"] == UNKNOWN]
        if state == UNKNOWN
        else []
    )
    return {"state": state, "components": traces, "next_checks": sorted(set(questions))}


def compare(profile, hidden=(), ablation=None):
    old, new = [evaluate(profile, v, hidden, ablation) for v in ("2022", "2023")]
    a, b = old["state"], new["state"]
    change = (
        "unresolved"
        if UNKNOWN in (a, b)
        else (
            "included"
            if (a, b) == (FAIL, PASS)
            else (
                "excluded"
                if (a, b) == (PASS, FAIL)
                else "retained" if a == PASS else "outside"
            )
        )
    )
    minimal = []
    if change in ("included", "excluded"):
        for size in (1, 2):
            for subset in combinations(("housing", "welfare"), size):
                if any(set(previous).issubset(subset) for previous in minimal):
                    continue
                mixed = {
                    k: "2023" if k in subset else "2022" for k in ("housing", "welfare")
                }
                if evaluate(profile, "2022", hidden, ablation, mixed)["state"] == b:
                    minimal.append(list(subset))
    return {"2022": old, "2023": new, "change": change, "minimal_rule_changes": minimal}
