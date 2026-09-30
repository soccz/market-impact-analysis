"""A label-free money view, with reversible character alignment to the source."""

from copy import deepcopy
from common import V3
from build_data import amounts
from model import prepare as original_prepare, predict as original_predict

PLACEHOLDER = "0원"


def project(mapping, start, end):
    assert 0 <= start < end <= len(mapping), (start, end, len(mapping))
    selected = mapping[start:end]
    return min(x[0] for x in selected), max(x[1] for x in selected)


def money_view(text, replacement=lambda span: PLACEHOLDER):
    """Only source text enters this transformation; roles/labels never enter it."""
    found = amounts(text)
    chars, forward, backward = [], [], []
    cursor = 0
    for sp in found:
        for i in range(cursor, sp["start"]):
            forward.append((len(chars), len(chars) + 1))
            backward.append((i, i + 1))
            chars.append(text[i])
        new = replacement(sp)
        assert new
        a, b = len(chars), len(chars) + len(new)
        forward.extend([(a, b)] * (sp["end"] - sp["start"]))
        backward.extend([(sp["start"], sp["end"])] * len(new))
        chars.extend(new)
        cursor = sp["end"]
    for i in range(cursor, len(text)):
        forward.append((len(chars), len(chars) + 1))
        backward.append((i, i + 1))
        chars.append(text[i])
    assert len(forward) == len(text) and len(backward) == len(chars)
    return dict(text="".join(chars), forward=forward, backward=backward, amounts=found)


def transform_row(row, view):
    """Project training/evaluation annotations only after constructing input text."""
    out = deepcopy(row)
    out["text"] = view["text"]
    for key in ["spans", "evidence"]:
        for span in out[key]:
            span["start"], span["end"] = project(
                view["forward"], span["start"], span["end"]
            )
            span["text"] = out["text"][span["start"] : span["end"]]
            if key == "spans":
                span["won"] = amounts(span["text"])[0]["won"]
    return out


def prepare(rows, tokenizer, normalized, max_length=128):
    if not normalized:
        return dict(
            rows=rows, encoded=original_prepare(rows, tokenizer, max_length), views=None
        )
    views = [money_view(r["text"]) for r in rows]
    transformed = [transform_row(r, v) for r, v in zip(rows, views)]
    encoded = original_prepare(transformed, tokenizer, max_length)
    return dict(rows=transformed, encoded=encoded, views=views)


def predict(model, rows, prepared, device):
    predictions = original_predict(model, prepared["rows"], prepared["encoded"], device)
    if prepared["views"] is None:
        return predictions
    for row, p, view in zip(rows, predictions, prepared["views"]):
        assert len(p["spans"]) == len(view["amounts"])
        restored = []
        for sp, original in zip(p["spans"], view["amounts"]):
            assert project(view["backward"], sp["start"], sp["end"]) == (
                original["start"],
                original["end"],
            )
            restored.append(
                dict(original, role=sp["role"], probabilities=sp["probabilities"])
            )
        p["spans"] = restored
        for ev in p["evidence"]:
            ev["start"], ev["end"] = project(view["backward"], ev["start"], ev["end"])
            ev["text"] = row["text"][ev["start"] : ev["end"]]
    return predictions
