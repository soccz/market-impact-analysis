"""Three-valued oracle for authored worlds, never supplied to the models."""


def all_of(values):
    if False in values:
        return False
    return None if None in values else True


def any_of(values):
    if True in values:
        return True
    return None if None in values else False


def label(value):
    return (
        "not_established" if value is None else "supported" if value else "contradicted"
    )


def oracle(family, facts):
    if family == "version_scope":
        limit = 100 if facts["old"] else 140 if facts["first"] else 110
        return facts["income"] <= limit
    if family == "only_exception":
        return all_of(
            [facts["education_only"], facts["credits"] >= 8, facts["residence"] >= 6]
        )
    if family == "threshold_or":
        checks = [facts["annual"] >= 1200, facts["monthly"] >= 100]
        return any_of(checks) if facts["first"] else all_of(checks)
    if family == "modality":
        if facts["state"] == "confirmed":
            return facts["amount"] == 50
        if facts["state"] == "denied" and facts["amount"] == 50:
            return False
        return None
    if family == "anchor_dates":
        return (
            facts["business_months"] >= 6
            if facts["target"] == "business"
            else facts["elapsed_months"] > 3
        )
    if family == "role_units":
        return None if facts["actual"] else facts["amount_won"] == 600000
    raise ValueError(family)
