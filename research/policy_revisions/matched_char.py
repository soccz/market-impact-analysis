"""Post-hoc input-matched character ablation for the frozen encoder probe."""

import argparse
import json
from pathlib import Path

from frozen_encoder import normalized
from model import amounts, classifier
from run import read, select_threshold, summarize


def candidate_context(text, bounds):
    a, b = bounds
    return text[max(0, a - 48) : a] + " 대상금액 " + text[b : b + 48]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not (args.out / "results.json").exists()
    args.out.mkdir(parents=True, exist_ok=True)
    train = read("train")
    relation, roles = classifier(), classifier()
    relation.fit(
        [normalized(r["text"])[0] for r in train], [r["relation"] for r in train]
    )
    xs, ys = [], []
    for r in train:
        if r["relation"] == "undetermined":
            continue
        text, spans = normalized(r["text"])
        for bounds, gold in zip(spans, r["spans"]):
            xs.append(candidate_context(text, bounds))
            ys.append(gold["role"])
    roles.fit(xs, ys)

    def predict(rows):
        output = []
        for row in rows:
            text, bounds = normalized(row["text"])
            p = relation.predict_proba([text])[0]
            candidates = amounts(row["text"])
            for span, b in zip(candidates, bounds):
                q = roles.predict_proba([candidate_context(text, b)])[0]
                span.update(
                    role=str(roles.classes_[q.argmax()]), confidence=float(q.max())
                )
            output.append(
                {
                    "relation": str(relation.classes_[p.argmax()]),
                    "confidence": float(p.max()),
                    "relation_probabilities": dict(
                        zip(map(str, relation.classes_), map(float, p))
                    ),
                    "candidates": candidates,
                }
            )
        return output

    dev = read("validation")
    raw = predict(dev)
    threshold, trace = select_threshold(dev, raw)
    (args.out / "selection.json").write_text(
        json.dumps({"threshold": threshold, "validation_candidates": trace}, indent=2)
        + "\n"
    )
    result = {
        "status": "post-hoc matched-input character ablation; not a new test",
        "threshold": threshold,
        "sets": {},
    }
    for split in ["validation", "evaluation", "official_cases"]:
        rows = read(split)
        ps = raw if split == "validation" else predict(rows)
        methods, scores = summarize(rows, ps, threshold)
        result["sets"][split] = scores
        (args.out / f"{split}_predictions.json").write_text(
            json.dumps(
                {"ids": [r["id"] for r in rows], "raw": ps, "methods": methods},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                s: {
                    m: {k: v for k, v in x.items() if k not in ["confusion"]}
                    for m, x in ms.items()
                    if m in ["span_model", "selective"]
                }
                for s, ms in result["sets"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
