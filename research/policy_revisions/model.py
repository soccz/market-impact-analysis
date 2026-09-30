"""Text-only relation and monetary-span models. Never accepts gold labels."""
import re
from decimal import Decimal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

MONEY = re.compile(r"(?<![\d.,])(?:\d[\d,]*(?:\.\d+)?\s*(?:조|억|천만|백만|만|천)\s*)*(?:\d[\d,]*(?:\.\d+)?\s*)?원")
PART = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(조|억|천만|백만|만|천)?")
MULTIPLIERS = {"조": 10**12, "억": 10**8, "천만": 10**7, "백만": 10**6, "만": 10**4, "천": 10**3, None: 1}


def amounts(text):
    results = []
    for match in MONEY.finditer(text):
        parts = list(PART.finditer(match[0][:-1]))
        if not parts:
            continue
        value = sum(Decimal(p[1].replace(",", "")) * MULTIPLIERS[p[2]] for p in parts)
        results.append({"start": match.start(), "end": match.end(), "text": match[0], "won": format(value.normalize(), "f")})
    return results


def mask(text):
    for span in reversed(amounts(text)):
        text = text[:span["start"]] + " 금액 " + text[span["end"]:]
    return re.sub(r"\d+", "숫자", text)


def context(text, span):
    left = mask(text[:span["start"]])[-48:]
    right = mask(text[span["end"]:])[:48]
    return left + " 대상금액 " + right


def classifier():
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=10, max_iter=1000, solver="lbfgs", random_state=731))


class SpanModel:
    def __init__(self):
        self.relation = classifier()
        self.roles = classifier()

    def fit(self, rows):
        self.relation.fit([mask(r["text"]) for r in rows], [r["relation"] for r in rows])
        samples = [(r, s) for r in rows if r["relation"] != "undetermined" for s in r["spans"]]
        self.roles.fit([context(r["text"], s) for r, s in samples], [s["role"] for _, s in samples])

    def predict(self, texts):
        relation_probs = self.relation.predict_proba([mask(t) for t in texts])
        relations = self.relation.classes_
        result = []
        for text, prob in zip(texts, relation_probs):
            relation = str(relations[prob.argmax()])
            spans = amounts(text)
            if spans:
                role_probs = self.roles.predict_proba([context(text, s) for s in spans])
                for span, p in zip(spans, role_probs):
                    span["role"] = str(self.roles.classes_[p.argmax()])
                    span["confidence"] = float(p.max())
            result.append({"relation": relation, "relation_probabilities": dict(zip(map(str, relations), map(float, prob))),
                           "confidence": float(prob.max()), "candidates": spans})
        return result


def ordered(raw):
    out = dict(raw)
    out["candidates"] = [{**s, "role": ("previous" if i == 0 else "current" if i == 1 else "other")} for i, s in enumerate(raw["candidates"])]
    return out


def keyword(text):
    spans = amounts(text)
    # Intentionally simple lexical baseline; does not model negation or scope.
    relation = "undetermined"
    if re.search(r"단위|환산", text):
        relation = "equivalent"
    elif re.search(r"정정|오기|오류|잘못", text):
        relation = "correction"
    elif re.search(r"변경|조정|인상|늘렸|바꾼", text):
        relation = "policy_change"
    return ordered({"relation": relation, "confidence": 1.0, "candidates": spans})


def decision(raw, constrained=False, threshold=0.0):
    relation = raw["relation"]
    args = [] if relation == "undetermined" else [s for s in raw["candidates"] if s["role"] != "other"]
    score = raw["confidence"]
    reason = None
    if constrained and relation != "undetermined":
        old = [s for s in args if s["role"] == "previous"]
        new = [s for s in args if s["role"] == "current"]
        if len(old) != 1 or len(new) != 1:
            reason = "argument_cardinality"
        elif (old[0]["won"] == new[0]["won"]) != (relation == "equivalent"):
            reason = "value_relation_inconsistent"
        score = min([score] + [s.get("confidence", 1.0) for s in args])
    if score < threshold:
        reason = reason or "below_validation_threshold"
    return {"relation": relation, "arguments": args, "accepted": reason is None, "abstention_reason": reason, "score": score}


def signature(spans):
    return sorted((s["start"], s["end"], s["role"], s["won"]) for s in spans if s["role"] != "other")


def correct(row, pred):
    return pred["relation"] == row["relation"] and signature(pred["arguments"]) == signature(row["spans"])
