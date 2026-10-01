"""Post-result, label-free sentence retrieval baseline; no new model fitting."""

from copy import deepcopy
import re
from common import V3
from build_data import amounts
from binding import prepare as full_prepare, predict as full_predict


def retrieve(query, text):
    boundaries = list(re.finditer(r"(?<=[.!?])\s+", text))
    starts = [0] + [m.end() for m in boundaries]
    ends = [m.start() for m in boundaries] + [len(text)]
    selected = [(a, b) for a, b in zip(starts, ends) if query and query in text[a:b]]
    chars = []
    back = []
    for a, b in selected:
        if chars:
            chars.append(" ")
            back.append(None)
        chars.extend(text[a:b])
        back.extend(range(a, b))
    result = "".join(chars)
    return dict(
        text=result, back=back, intervals=selected, answerable=bool(amounts(result))
    )


def source_parts(back, a, b):
    groups = []
    for index in back[a:b]:
        if index is None:
            continue
        if groups and index == groups[-1][1]:
            groups[-1][1] = index + 1
        else:
            groups.append([index, index + 1])
    return groups


def prepare(rows, tokenizer, max_length=192):
    views = [retrieve(r["query"], r["text"]) for r in rows]
    ids = [i for i, v in enumerate(views) if v["answerable"]]
    routed = []
    for i in ids:
        text = views[i]["text"]
        # Dummy labels satisfy the old tensor API; prediction never uses them.
        routed.append(
            dict(
                id=rows[i]["id"],
                query=rows[i]["query"],
                text=text,
                relation="unspecified",
                state="unspecified",
                evidence=[],
                spans=[dict(s, role="other") for s in amounts(text)],
            )
        )
    encoded = full_prepare(routed, tokenizer, "plain", max_length) if routed else None
    return dict(views=views, ids=ids, rows=routed, encoded=encoded)


def predict(model, rows, prepared, device):
    ps = (
        full_predict(model, prepared["rows"], prepared["encoded"], device)
        if prepared["rows"]
        else []
    )
    mapped = dict(zip(prepared["ids"], ps))
    out = []
    for i, (row, view) in enumerate(zip(rows, prepared["views"])):
        record = dict(
            id=row["id"],
            answerable=view["answerable"],
            selected_intervals=view["intervals"],
            reason=(
                "query-containing sentences include a parsed amount"
                if view["answerable"]
                else "no parsed amount in query-containing sentences"
            ),
            prediction=None,
        )
        if i in mapped:
            p = deepcopy(mapped[i])
            spans = {}
            for s in p["spans"]:
                loc = source_parts(view["back"], s["start"], s["end"])
                assert len(loc) == 1
                a, b = loc[0]
                s.update(start=a, end=b, text=row["text"][a:b])
                spans[(a, b)] = s
            restored = []
            for s in amounts(row["text"]):
                key = (s["start"], s["end"])
                restored.append(
                    spans[key]
                    if key in spans
                    else dict(s, role="other", probabilities=[1.0, 0.0, 0.0])
                )
            ev = []
            for s in p["evidence"]:
                for a, b in source_parts(view["back"], s["start"], s["end"]):
                    ev.append(
                        dict(start=a, end=b, text=row["text"][a:b], kind=s["kind"])
                    )
            p.update(spans=restored, evidence=ev)
            record["prediction"] = p
        out.append(record)
    return out
