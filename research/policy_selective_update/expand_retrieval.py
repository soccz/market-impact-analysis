"""Exploratory retrieval repair: matched chunk budgets, no labels in selection."""

import argparse
import itertools
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from core import sha
from infer import ROOT
from retrieve import chunks, source_covered
from prepare_inputs import canonical


def choose(claim, candidates, mode):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False)
    matrix = vec.fit_transform([c["text"] for c in candidates])
    scores = (matrix @ vec.transform([claim]).T).toarray().ravel()
    ranked = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
    if mode == "top5":
        ids = ranked[:5]
    elif mode == "neighbor5":
        ids = ranked[:3]
        for i in ranked[:3]:
            for j in [i - 1, i + 1]:
                if (
                    0 <= j < len(candidates)
                    and candidates[j]["page"] == candidates[i]["page"]
                    and j not in ids
                ):
                    ids.append(j)
                    if len(ids) == 5:
                        break
            if len(ids) == 5:
                break
        # No padding from unrelated chunks if no neighbor exists.
    else:
        raise ValueError(mode)
    return [{**candidates[i], "retrieval_score": float(scores[i])} for i in ids]


def build(cache, mode):
    cache = Path(cache)
    original = json.loads((ROOT / "curated_reference.json").read_text())
    source = json.loads((ROOT / "sources.json").read_text())
    pages = {s["id"]: canonical(cache / (s["id"] + ".pdf")) for s in source}
    bs = {b["id"]: b for b in original["bundles"]}
    data = {"bundles": [], "cases": []}
    public = {"bundles": [], "cases": []}
    for c in original["cases"]:
        b = bs[c["bundle"]]
        row = json.loads(json.dumps(c))
        row["bundle"] = c["id"]
        row["retrieval_coverage"] = {}
        row["required_source_spans"] = {}
        nb = {"id": c["id"], "kind": mode}
        pb = {**nb, "source_locators": {}}
        for phase, prefix in [("before", "O"), ("after", "N")]:
            loc = b["source_locators"][phase]
            doc = next(iter(loc.values()))["document"]
            selected = choose(c["claim"], chunks(pages[doc], prefix), mode)
            nb[phase] = {x["id"]: x["text"] for x in selected}
            pb["source_locators"][phase] = {
                x["id"]: {k: v for k, v in x.items() if k not in ["text", "id"]}
                | {"document": doc, "text_sha256": sha(x["text"].encode())}
                for x in selected
            }
            required = [loc[i] for i in c["reference"][phase + "_evidence"]]
            row["required_source_spans"][phase] = required
            row["retrieval_coverage"][phase] = all(
                source_covered(selected, r) for r in required
            )
            sufficient = None
            for n in range(1, len(selected) + 1):
                for subset in itertools.combinations(selected, n):
                    if all(source_covered(subset, r) for r in required):
                        sufficient = [x["id"] for x in subset]
                        break
                if sufficient:
                    break
            row["reference"][phase + "_evidence"] = sufficient or [
                "MISSING_FULL_SOURCE_SUPPORT"
            ]
        data["bundles"].append(nb)
        public["bundles"].append(pb)
        data["cases"].append(row)
        public["cases"].append(row)
    return data, public


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    for mode in ["top5", "neighbor5"]:
        data, public = build(a.cache, mode)
        for path, value in [
            (Path(a.cache) / (mode + "_inputs.json"), data),
            (ROOT / (mode + "_reference.json"), public),
        ]:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        print(
            mode,
            {
                p: sum(c["retrieval_coverage"][p] for c in data["cases"])
                for p in ["before", "after"]
            },
        )
