"""Independently fitted character n-gram axes and local monetary-role classifier."""

import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from build_data import RELATIONS, STATES, amounts
from model import ROLES
from evaluate import read

ROOT = Path(__file__).resolve().parent


def clf():
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2),
        LogisticRegression(C=1, max_iter=2000, random_state=17),
    )


def text(r):
    return r["query"] + " [SEP] " + r["text"]


def local(r, sp):
    return (
        r["query"]
        + " [SEP] "
        + r["text"][max(0, sp["start"] - 48) : sp["start"]]
        + " [AMOUNT] "
        + r["text"][sp["end"] : sp["end"] + 48]
    )


def probs(model, texts, labels):
    p = model.predict_proba(texts)
    return [
        [float(row[list(model.classes_).index(label)]) for label in labels] for row in p
    ]


def main():
    out = ROOT / "results/character"
    assert not out.exists()
    out.mkdir(parents=True)
    train = read("train")
    axes = {
        key: clf().fit([text(r) for r in train], [r[key] for r in train])
        for key in ["relation", "state"]
    }
    role = clf().fit(
        [local(r, s) for r in train for s in r["spans"]],
        [s["role"] for r in train for s in r["spans"]],
    )
    for name in [
        "validation",
        "evaluation",
        "distractor",
        "units",
        "evidence_removed",
        "official",
    ]:
        rows = read(name)
        rp = probs(axes["relation"], [text(r) for r in rows], RELATIONS)
        sp = probs(axes["state"], [text(r) for r in rows], STATES)
        results = []
        for i, r in enumerate(rows):
            spans = amounts(r["text"])
            if spans:
                for s, p in zip(
                    spans, probs(role, [local(r, s) for s in spans], ROLES)
                ):
                    s.update(role=ROLES[int(np.argmax(p))], probabilities=p)
            results.append(
                dict(
                    id=r["id"],
                    relation=RELATIONS[int(np.argmax(rp[i]))],
                    state=STATES[int(np.argmax(sp[i]))],
                    relation_probabilities=rp[i],
                    state_probabilities=sp[i],
                    spans=spans,
                    evidence=[],
                    joint_probabilities=None,
                )
            )
        (out / (name + "_predictions.json")).write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
