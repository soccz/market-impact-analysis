"""Same passage, two incompatible target queries. Authored probes, not human gold."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from common import ROOT, save
from build_data import EVENTS, STATUS, RELATIONS, STATES, amounts

NAMES = {
    "train": [
        ("가온주거비", "새봄돌봄비"),
        ("다온정착금", "누리활동금"),
        ("마루교육비", "초롱양육비"),
        ("한빛지원금", "솔빛장려금"),
    ],
    "validation": [("바른훈련비", "온유교통비"), ("푸른생활금", "다솜급식비")],
    "evaluation": [
        ("가람정착금", "별빛수당"),
        ("산들주거비", "해솔돌봄비"),
        ("여울교육비", "새론활동금"),
        ("소담지원금", "한울장려금"),
    ],
}


def make(split, family, frame, case, layout="blocks", reverse=False, won=False):
    names = NAMES[split][frame]
    c0 = case % 24
    shifts = [1, 7, 11, 13, 19] if split == "train" else [5, 17]
    c1 = (c0 + shifts[case // 24]) % 24
    labels = [divmod(c0, 6), divmod(c1, 6)]
    order = [0, 1] if (case + frame) % 2 == 0 else [1, 0]
    if reverse:
        order.reverse()
    # Amount magnitude and direction are not globally tied to target identity/order.
    values = []
    for j, (r, _) in enumerate(labels):
        h = int(
            hashlib.sha256(f"{split}/{frame}/{case}/{j}".encode()).hexdigest()[:8], 16
        )
        before = 100 + h % 800
        after = before if r == 2 else before + (17 if h % 2 else -23)
        values.append((before, after))
    parts, spans, evidence = [], [], []

    def add(text, owner=None, kind=None, role=None):
        start = sum(map(len, parts))
        parts.append(text)
        if role:
            found = amounts(text)
            assert len(found) == 1
            spans.append(
                dict(
                    found[0], start=start, end=start + len(text), owner=owner, role=role
                )
            )
        if kind:
            evidence.append(
                dict(
                    start=start,
                    end=start + len(text),
                    text=text,
                    owner=owner,
                    kind=kind,
                )
            )

    def cash(j, role):
        value = values[j][0 if role == "before" else 1]
        add(f"{value * 10000:,}원" if won else f"{value}만 원", owner=j, role=role)

    def numbers(j, prefix):
        add(prefix)
        if frame % 2:
            add("후속안 금액은 ")
            cash(j, "after")
            add(", 기준 문서의 금액은 ")
            cash(j, "before")
        else:
            add("기준 문서의 금액은 ")
            cash(j, "before")
            add(", 후속안 금액은 ")
            cash(j, "after")
        add("이다. ")

    def event(j, prefix):
        r, s = labels[j]
        add(prefix)
        add(EVENTS[family][r], owner=j, kind="relation")
        add(" 사안은 ")
        add(STATUS[family][s], owner=j, kind="state")
        add(" ")

    if layout == "anaphora":
        add(f"{names[order[0]]} 및 {names[order[1]]}, 두 사업을 차례로 설명한다. ")
        for position, j in enumerate(order):
            ref = "앞서 언급한 사업" if position == 0 else "뒤에 언급한 사업"
            numbers(j, ref + "의 ")
            event(j, ref + "에 대한 ")
    elif layout == "interleaved":
        for j in order:
            numbers(j, names[j] + "의 ")
        for j in reversed(order):
            event(j, names[j] + "에 대한 ")
    else:
        for j in order:
            if frame < 2:
                numbers(j, names[j] + "의 ")
                event(j, names[j] + "에 대한 ")
            else:
                event(j, names[j] + "에 대한 ")
                numbers(j, names[j] + "의 ")
    text = "".join(parts).rstrip()
    doc = f"{split}-{family}-{frame}-{case}"
    rows = []
    for j, name in enumerate(names):
        r, s = labels[j]
        rows.append(
            dict(
                id=f"{layout}{'-reverse' if reverse else ''}{'-won' if won else ''}/{doc}/q{j}",
                document_id=doc,
                query=name,
                text=text,
                relation=RELATIONS[r],
                state=STATES[s],
                spans=[
                    {
                        k: (
                            (sp["role"] if sp["owner"] == j else "other")
                            if k == "role"
                            else v
                        )
                        for k, v in sp.items()
                        if k != "owner"
                    }
                    for sp in spans
                ],
                evidence=[
                    {k: v for k, v in e.items() if k != "owner"}
                    for e in evidence
                    if e["owner"] == j
                ],
                owner_evidence=deepcopy(evidence),
                entities=names,
                target=j,
                position=order.index(j),
                family=family,
                frame=frame,
                layout=layout,
            )
        )
    return rows


def main():
    assert not (ROOT / "freeze.json").exists(), "Do not regenerate frozen data"
    sets = {}
    for split, families, count in [
        ("train", range(4), 120),
        ("validation", [6, 7], 48),
        ("evaluation", range(8, 12), 48),
    ]:
        sets[split] = [
            row
            for frame, family in enumerate(families)
            for case in range(count)
            for row in make(split, family, frame, case)
        ]
    for name in ["interleaved", "anaphora", "order", "units"]:
        sets[name] = [
            row
            for frame, family in enumerate(range(8, 12))
            for case in range(48)
            for row in make(
                "evaluation",
                family,
                frame,
                case,
                layout=name if name in ["interleaved", "anaphora"] else "blocks",
                reverse=name == "order",
                won=name == "units",
            )
        ]
    sets["absent"] = []
    for old in sets["evaluation"][::2]:
        row = deepcopy(old)
        row.update(
            id="absent/" + old["document_id"],
            query="미등장급여",
            target=None,
            position=None,
            relation="unspecified",
            state="unspecified",
            evidence=[],
        )
        for s in row["spans"]:
            s["role"] = "other"
        sets["absent"].append(row)
    audit = {}
    for name, rows in sets.items():
        p = ROOT / "data" / (name + ".jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        audit[name] = dict(
            rows=len(rows),
            passages=len({r["text"] for r in rows}),
            labels=dict(Counter(r["relation"] + "/" + r["state"] for r in rows)),
        )
    save(
        ROOT / "data/design.json",
        dict(
            sets=audit,
            authorship="AI-assisted authored grammar; human reviews 0",
            note="Two query rows share a passage and must stay in one split. All 24 combinations trained; no composition holdout claim. Evaluation uses new names, lexical families and event-pair shifts. Interleaved/anaphora rewrite the SAME evaluation events.",
        ),
    )
    print({n: len(r) for n, r in sets.items()})


if __name__ == "__main__":
    main()
