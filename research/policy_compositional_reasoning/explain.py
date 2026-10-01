"""Post-hoc factual witnesses and minimal sufficient fact sets for a given program.

These explain only the extracted rule. They are not evidence of source faithfulness.
"""

import itertools
from logic import policy, profile, evaluate
from worlds import candidates, execute, MAX_WORLDS


def explain(p, q, allowed):
    clauses, ps = policy(p, allowed)
    values, pol, qs = profile(q)
    if clauses is None or values is None:
        return {"status": "unavailable", "policy_status": ps, "profile_status": qs}
    result, _, influential, mode, count = execute(clauses, values)
    relevant = sorted({a["field"] for c in clauses for a in c})
    known = {k: v for k, v in values.items() if k in relevant and v is not None}
    if mode == "budget_fallback":
        return {"status": "budget_fallback", "candidate_fields": influential}
    if result is not None:
        if len(known) > 10:
            return {"status": "fact_subset_budget"}
        minimal = []
        for size in range(len(known) + 1):
            for keys in itertools.combinations(sorted(known), size):
                small = {k: known[k] for k in keys}
                if execute(clauses, small)[0] is result:
                    minimal.append(small)
            if minimal:
                break
        return {
            "status": "determinate",
            "condition_value": result,
            "minimal_fact_sets": minimal,
            "interpretation": "minimum-cardinality observed fact subsets under this extracted program only",
        }
    missing = [f for f in relevant if f not in known]
    ds = [candidates(f, clauses) for f in missing]
    n = 1
    for d in ds:
        n *= len(d)
    if n > MAX_WORLDS:
        return {"status": "budget_fallback", "candidate_fields": influential}
    witnesses = {}
    for vals in itertools.product(*ds):
        completion = dict(zip(missing, vals))
        v = evaluate(clauses, known | completion)[0]
        key = "passes" if v else "fails"
        if key not in witnesses:
            witnesses[key] = completion
        if len(witnesses) == 2:
            break
    return {
        "status": "underdetermined",
        "candidate_fields": influential,
        "witnesses": witnesses,
        "interpretation": "hypothetical completions of unknown fields, not observed applicant data",
    }
