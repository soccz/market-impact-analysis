"""Auxiliary span coverage accepts equivalent overlapping chunk citations."""

from retrieve import source_covered


def covered(locators, cited_ids, required):
    cited = [locators[i] for i in cited_ids if i in locators]
    return all(
        source_covered([x for x in cited if x["document"] == loc["document"]], loc)
        for loc in required
    )


def evaluate(reference, scores):
    bundles = {b["id"]: b for b in reference["bundles"]}
    cases = {r["id"]: r for r in reference["cases"]}
    result = {}
    for model, methods in scores.items():
        rows = []
        for row in methods["refresh_all"]["cases"]:
            r = cases[row["id"]]
            b = bundles[r["bundle"]]
            item = {"id": r["id"]}
            for phase, stage in [("before", "old"), ("after", "fresh")]:
                locations = b["source_locators"][phase]
                required = r.get("required_source_spans", {}).get(phase)
                if required is None:
                    required = [
                        locations[i] for i in r["reference"][phase + "_evidence"]
                    ]
                supply = covered(locations, list(locations), required)
                evidence = covered(locations, row[stage]["evidence"], required)
                correct = row[stage]["decision"] == r["reference"][phase]
                item[phase] = {
                    "supplied_full_span": supply,
                    "cited_full_span": evidence,
                    "correct": correct,
                    "joint": correct and evidence,
                }
            rows.append(item)
        result[model] = {
            "cases": rows,
            "summary": {
                phase: {
                    k: sum(r[phase][k] for r in rows)
                    for k in [
                        "supplied_full_span",
                        "cited_full_span",
                        "correct",
                        "joint",
                    ]
                }
                for phase in ["before", "after"]
            },
        }
    return result
