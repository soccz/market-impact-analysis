"""Author-controlled compositional probes, not an independently collected benchmark."""

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELATIONS = ["correction", "policy_change", "restatement", "unspecified"]
STATES = [
    "asserted",
    "planned",
    "under_review",
    "denied",
    "hypothetical",
    "unspecified",
]
HELD = {(0, 1), (1, 3), (2, 4), (3, 2)}
PATTERN = re.compile(
    r"(?:\d[\d,]*(?:\.\d+)?\s*(?:(?:천|백|십)?(?:조|억|만)|천)\s*)+(?:\d[\d,]*(?:\.\d+)?\s*)?원|\d[\d,]*(?:\.\d+)?\s*원"
)
MULT = {
    "조": 10**12,
    "억": 10**8,
    "천만": 10**7,
    "백만": 10**6,
    "십만": 10**5,
    "만": 10**4,
    "천": 1000,
    "": 1,
}

MULT.update(
    {
        prefix + unit: factor * base
        for prefix, factor in [("십", 10), ("백", 100), ("천", 1000)]
        for unit, base in [("만", 10**4), ("억", 10**8), ("조", 10**12)]
    }
)

# Each row is a surface family. Splits do not share rows.
EVENTS = [
    [
        "잘못 기재한 금액의 정정",
        "실제 지원 규모의 변경",
        "금액은 같고 단위만 다른 재표기",
        "관계 설명 없는 두 금액의 병기",
    ],
    [
        "입력 오류를 바로잡는 수정",
        "지원 액수를 실질적으로 조정하는 개편",
        "동일한 액수를 다른 단위로 쓰는 작업",
        "이유가 제시되지 않은 금액 나열",
    ],
    [
        "오탈자로 생긴 금액 착오의 교정",
        "실제 지급액을 바꾸는 조치",
        "액수 변화 없이 단위만 환산하는 표기",
        "원인을 밝히지 않은 수치의 동시 기재",
    ],
    [
        "계산 실수를 바로잡는 정정",
        "실질적인 예산 규모 조정",
        "같은 돈을 다른 단위로 적는 변경",
        "사유를 적지 않은 두 액수의 제시",
    ],
    [
        "오기된 수치의 바로잡기",
        "실제 지원 금액을 변경하는 결정",
        "가치가 같은 금액의 단위 환산",
        "두 수치 사이의 관계가 설명되지 않은 기재",
    ],
    [
        "기재 착오에 대한 정정 작업",
        "사업의 실제 지급 규모 변경",
        "실질 금액을 유지한 단위 표기 전환",
        "금액 차이의 이유가 없는 나열",
    ],
    [
        "문서상의 숫자 오류를 고치는 일",
        "지원 총액의 실질적 개편",
        "동일 금액의 표현 단위 교체",
        "연관성 설명 없이 액수를 함께 적는 일",
    ],
    [
        "착오로 적힌 액수의 수정",
        "지급 규모 자체의 조정",
        "환산에 따른 동액 재기재",
        "사유 미기재 상태의 금액 병렬 제시",
    ],
    [
        "잘못 산출된 공고 금액을 바로잡는 조치",
        "지급할 돈의 규모 자체를 바꾸는 조치",
        "실질 가치의 변화 없이 환산 단위만 바꾸는 조치",
        "왜 다른지 밝히지 않은 두 액수의 동시 표기",
    ],
    [
        "오류가 있는 숫자를 고쳐 적는 작업",
        "정책상 지원 수준을 실질적으로 바꾸는 작업",
        "원래 액수와 똑같은 값을 다른 단위로 적는 작업",
        "두 숫자의 연관성을 제시하지 않는 작성",
    ],
    [
        "문서에 잘못 들어간 액수의 교체",
        "수혜자가 받을 지원 규모의 실질적인 조정",
        "내용상 금액 차이가 없는 단위 재표현",
        "금액 사이의 관계를 밝히지 않은 기록",
    ],
    [
        "산출 과정의 실수를 수정하는 정정",
        "사업비의 실질적인 증감",
        "동일 액수의 환산 표기",
        "원인 설명이 빠진 두 액수의 기재",
    ],
]
STATUS = [
    [
        "이미 확정되었다.",
        "추후 추진할 계획이다.",
        "현재 검토 중이며 아직 정해지지 않았다.",
        "사실이 아니며 이루어지지 않았다.",
        "실제로 발생했다고 가정한 예시일 뿐이다.",
        "진행 여부가 문서에 적혀 있지 않다.",
    ],
    [
        "결정이 내려져 완료되었다.",
        "앞으로 진행할 예정이다.",
        "논의하고 있지만 확정 전이다.",
        "실시했다는 주장은 사실과 다르다.",
        "만약 이루어진다면 어떨지 상정한 상황이다.",
        "상태에 관한 설명이 생략되어 있다.",
    ],
    [
        "확정된 조치라고 발표했다.",
        "향후 시행하겠다는 계획이다.",
        "협의 중이고 결론은 나오지 않았다.",
        "진행한 적이 없다고 명시했다.",
        "현실의 조치가 아니라 조건부 가정이다.",
        "확정인지 아닌지 언급하지 않았다.",
    ],
    [
        "실행을 마쳤다고 밝혔다.",
        "다음 단계에 시행할 방침이다.",
        "검토 단계로 결정되지 않았다.",
        "실행됐다는 보도를 부인했다.",
        "성립한다고 가정할 때의 예다.",
        "현재 어떤 단계인지 기재하지 않았다.",
    ],
    [
        "결정된 사항으로 공표되었다.",
        "앞으로 실행하려는 계획에 포함됐다.",
        "심의 중이며 확정된 바 없다.",
        "이루어졌다는 설명은 잘못된 것이다.",
        "발생을 전제로 한 가상 상황이다.",
        "추진 상태가 제시되어 있지 않다.",
    ],
    [
        "이미 시행한 조치라고 설명했다.",
        "나중에 추진할 예정이라고 밝혔다.",
        "논의 단계에 머물러 결정 전이다.",
        "실시되었다는 내용을 명시적으로 부인했다.",
        "가정상 이루어지는 경우를 말한다.",
        "진행 단계에 대해서는 아무 말이 없다.",
    ],
    [
        "최종 결정되어 반영되었다.",
        "향후 실행하는 것을 목표로 세웠다.",
        "협의는 계속되고 최종안은 정해지지 않았다.",
        "그런 조치는 없었다고 설명했다.",
        "만일 시행될 경우라는 조건에서만 성립한다.",
        "그 상태는 알 수 없도록 생략되어 있다.",
    ],
    [
        "확정 사실로 안내했다.",
        "장래에 이행할 계획을 제시했다.",
        "아직 심사 중으로 확정하지 않았다.",
        "시행했다는 주장에 대해 사실무근이라고 밝혔다.",
        "현실의 사실이 아닌 가상의 전제이다.",
        "현재 상태를 판독할 문구가 없다.",
    ],
    [
        "결정을 끝내고 이행한 사항이라고 알렸다.",
        "앞으로 이루어질 조치로 예고했다.",
        "아직 합의하지 못해 논의를 이어가고 있다.",
        "그러한 일이 있었다는 것은 사실이 아니라고 알렸다.",
        "가상으로 발생한 상황을 전제한 설명이다.",
        "이행 여부를 판단할 정보는 제공하지 않는다.",
    ],
    [
        "확정하여 적용을 마쳤다는 설명이다.",
        "향후 추진하겠다는 의향을 담고 있다.",
        "논의가 끝나지 않아 확정 여부는 미정이다.",
        "실제로 그런 조치를 취하지 않았다는 설명이다.",
        "이루어졌다고 치는 조건부 시나리오이다.",
        "어느 단계에 있는지는 밝혀져 있지 않다.",
    ],
    [
        "결정이 완료된 사실로 기재되어 있다.",
        "실행에 옮길 장래의 방안이라고 적었다.",
        "당국과 협의하고 있어 아직 결정 전이라고 적었다.",
        "발생했다는 해석을 부인하는 내용이다.",
        "사실로 보고한 것이 아니라 가정을 설정한 것이다.",
        "결정이나 실행에 대한 문장은 없다.",
    ],
    [
        "최종 승인되어 시행을 마친 일이다.",
        "다음에 실시할 방안으로 제안되었다.",
        "검토가 진행되고 있지만 결론을 내리지 않았다.",
        "시행된 적이 없다는 입장을 명시했다.",
        "발생하는 경우를 상상한 예시이다.",
        "실제 진행 상태는 언급되지 않는다.",
    ],
]


def amounts(text):
    out = []
    for m in PATTERN.finditer(text):
        t = re.sub(r"\s+", "", m.group()).replace(",", "")
        value = sum(
            Decimal(n) * MULT[u]
            for n, u in re.findall(r"([\d.]+)((?:천|백|십)?(?:조|억|만)|천)?", t[:-1])
        )
        out.append(
            dict(start=m.start(), end=m.end(), text=m.group(), won=str(int(value)))
        )
    return out


def make(split, family, frame, r, s, mode="base"):
    k = {"train": 0, "validation": 1, "evaluation": 2}[split]
    q = ["가람", "누리", "마루", "다솜", "라온", "해솔", "아람", "도담"][frame] + [
        "지원금",
        "활동비",
        "정착금",
    ][k]
    before = 31 + 40 * k + frame * 3
    after = before if r == 2 or (r == 3 and frame % 2) else before + 17
    b = f"{before}만 원"
    a = f"{after*10000:,}원" if r == 2 else f"{after}만 원"
    d = f"{200+k*80+frame}만 원"
    event, state = EVENTS[family][r], STATUS[family][s]
    parts = [f"{q}에 관한 설명이다. "]
    spans = []
    evidence = []

    def add(t):
        parts.append(t)

    def amt(t, role):
        start = sum(map(len, parts))
        add(t)
        spans.append(
            dict(
                start=start,
                end=start + len(t),
                text=t,
                won=amounts(t)[0]["won"],
                role=role,
            )
        )

    if family % 2:
        add("후속안에 적은 액수는 ")
        amt(a, "after")
        add(", 기준 문서의 액수는 ")
        amt(b, "before")
        add("이다. ")
    else:
        add("기준 문서에는 ")
        amt(b, "before")
        add(", 후속안에는 ")
        amt(a, "after")
        add("이라고 적혀 있다. ")
    if family % 3 == 0:
        add("별개 사업인 별빛수당의 금액은 ")
        amt(d, "other")
        add("이다. ")
    add(f"{q}의 ")
    start = sum(map(len, parts))
    add(event)
    evidence.append(
        dict(start=start, end=start + len(event), kind="relation", text=event)
    )
    add(" 사안은 ")
    start = sum(map(len, parts))
    add(state)
    evidence.append(dict(start=start, end=start + len(state), kind="state", text=state))
    text = "".join(parts)
    if mode == "distractor":
        extra = " 한편 별빛수당은 710만 원에서 930만 원으로 오류 정정을 했다는 보도를 부인했다."
        spans += [
            dict(
                x, start=x["start"] + len(text), end=x["end"] + len(text), role="other"
            )
            for x in amounts(extra)
        ]
        text += extra
    if mode == "units":
        for sp in sorted(spans, key=lambda x: x["start"], reverse=True):
            oldend = sp["end"]
            new = f"{int(sp['won']):,}원"
            delta = len(new) - (sp["end"] - sp["start"])
            text = text[: sp["start"]] + new + text[sp["end"] :]
            for x in spans + evidence:
                if x is sp:
                    continue
                if x["start"] >= oldend:
                    x["start"] += delta
                    x["end"] += delta
            sp["end"] = sp["start"] + len(new)
            sp["text"] = new
    if mode == "evidence_removed":
        for ev in reversed(evidence):
            text = text[: ev["start"]] + "[설명 생략]" + text[ev["end"] :]
        evidence = []
    key = f"{split}-{family}-{frame}-{r}-{s}"
    return dict(
        id=hashlib.sha256((key + "-" + mode).encode()).hexdigest()[:16],
        pair_id=key,
        split=split,
        family=family,
        frame=frame,
        query=q,
        text=text,
        relation=RELATIONS[r],
        state=STATES[s],
        spans=spans,
        evidence=evidence,
        composition="held_out" if (r, s) in HELD else "seen",
        condition=mode,
        label_status="AI-authored provisional synthetic",
    )


def write(name, rows):
    (ROOT / "data" / f"{name}.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    )


def main():
    assert not (
        ROOT / "freeze.json"
    ).exists(), "Frozen data cannot be regenerated in place"
    manifest = {}
    alltext = set()
    for split, families, frames in [
        ("train", range(6), 8),
        ("validation", range(6, 8), 4),
        ("evaluation", range(8, 12), 6),
    ]:
        rows = [
            make(split, f, i, r, s)
            for f in families
            for i in range(frames)
            for r in range(4)
            for s in range(6)
            if split == "evaluation" or (r, s) not in HELD
        ]
        assert all(x["text"] not in alltext for x in rows)
        alltext.update(x["text"] for x in rows)
        normalized = {
            re.sub(r"\d[\d,]*", "#", x["text"].replace(x["query"], "TARGET"))
            for x in rows
        }
        manifest[split] = {
            "n": len(rows),
            "families": list(families),
            "frames": frames,
            "normalized_strings": len(normalized),
            "held_out": sum(x["composition"] == "held_out" for x in rows),
        }
        write(split, rows)
    for mode in ["distractor", "units", "evidence_removed"]:
        rows = [
            make("evaluation", f, i, r, s, mode)
            for f in range(8, 12)
            for i in range(6)
            for r in range(4)
            for s in range(6)
        ]
        write(mode, rows)
    (ROOT / "data/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
