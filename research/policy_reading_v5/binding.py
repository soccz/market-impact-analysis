"""Mark an exact query mention without consulting gold roles, evidence or entities."""

from copy import deepcopy
import re
from common import V3, V4
from build_data import amounts
from representation import project, transform_row
from model import prepare as original_prepare, predict as original_predict


def input_view(query, text, condition):
    mode = condition.removeprefix("legacy_")
    changes = [(s["start"], s["end"], "0원") for s in amounts(text)]
    if mode == "marked":
        assert query
        changes += [
            (m.start(), m.end(), "지정사업")
            for m in re.finditer(re.escape(query), text)
        ]
    changes.sort()
    chars, forward, backward = [], [], []
    cursor = 0
    for a, b, new in changes:
        assert a >= cursor, "Overlapping query and monetary expression unsupported"
        for i in range(cursor, a):
            forward.append((len(chars), len(chars) + 1))
            backward.append((i, i + 1))
            chars.append(text[i])
        left = len(chars)
        chars.extend(new)
        forward.extend([(left, len(chars))] * (b - a))
        backward.extend([(a, b)] * len(new))
        cursor = b
    for i in range(cursor, len(text)):
        forward.append((len(chars), len(chars) + 1))
        backward.append((i, i + 1))
        chars.append(text[i])
    return dict(
        query="지정사업" if mode in ["marked", "blind"] else query,
        text="".join(chars),
        forward=forward,
        backward=backward,
        amounts=amounts(text),
    )


def prepare(rows, tokenizer, condition, max_length=192):
    views = [input_view(r["query"], r["text"], condition) for r in rows]
    transformed = [transform_row(r, v) for r, v in zip(rows, views)]
    for r, v in zip(transformed, views):
        r["query"] = v["query"]
    return dict(
        rows=transformed,
        views=views,
        encoded=original_prepare(transformed, tokenizer, max_length),
    )


def predict(model, rows, prepared, device):
    ps = original_predict(model, prepared["rows"], prepared["encoded"], device)
    for row, p, view in zip(rows, ps, prepared["views"]):
        assert len(p["spans"]) == len(view["amounts"])
        restored = []
        for sp, orig in zip(p["spans"], view["amounts"]):
            assert project(view["backward"], sp["start"], sp["end"]) == (
                orig["start"],
                orig["end"],
            )
            restored.append(
                dict(orig, role=sp["role"], probabilities=sp["probabilities"])
            )
        p["spans"] = restored
        for ev in p["evidence"]:
            ev["start"], ev["end"] = project(view["backward"], ev["start"], ev["end"])
            ev["text"] = row["text"][ev["start"] : ev["end"]]
    return ps
