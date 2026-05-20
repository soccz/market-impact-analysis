#!/usr/bin/env python3
"""
AI가 정리한 39건 제안 표를 review_B_39_input.csv에 채워넣음.

룰:
- 자동 다수결과 같은 값은 빈칸 유지
- 다른 값만 your_relevance / your_stance에 기록
- #36 (거래세 인하)는 가격 조건부 → 자동값 유지 (neutral)
- your_note: "AI-assisted suggestion; verify before apply"

이후 본인이 review_B_39.html과 csv를 같이 열고 한 행씩 확인해야 정석.
"""
import csv


# AI 제안 표 (사용자 메시지에서 추출)
# 컬럼: (idx, suggest_rel, suggest_stance) — None이면 자동값 유지
SUGGESTIONS = {
    1:  ("1", "support"),
    2:  ("1", "contradict"),
    3:  ("2", "contradict"),
    4:  None,  # rel=1, stance=contradict (자동값과 같음)
    5:  None,
    6:  ("1", "contradict"),
    7:  None,
    8:  ("1", "support"),
    9:  ("1", "support"),
    10: ("1", "contradict"),
    11: None,
    12: None,
    13: ("1", "contradict"),
    14: ("0", "neutral"),     # rel 변경
    15: ("1", "neutral"),
    16: ("0", "neutral"),     # rel 변경
    17: ("1", "support"),
    18: ("1", "contradict"),
    19: None,
    20: None,
    21: None,
    22: ("1", "contradict"),
    23: None,
    24: None,
    25: None,
    26: ("1", "support"),
    27: ("1", "support"),
    28: None,
    29: ("1", "contradict"),
    30: None,
    31: None,
    32: ("1", "support"),
    33: None,
    34: None,
    35: ("1", "contradict"),
    36: None,                 # 가격 조건부 → 자동값(neutral) 유지
    37: ("1", "contradict"),
    38: ("2", "support"),     # rel 변경
    39: None,
}

INPUT = "review_B_39_input.csv"
OUTPUT = "review_B_39_input.csv"
NOTE = "AI-assisted suggestion; verify before apply"


def main():
    with open(INPUT, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    if len(rows) != 39:
        raise SystemExit(f"Expected 39 rows, got {len(rows)}")

    changes = []
    for idx, row in enumerate(rows, start=1):
        sug = SUGGESTIONS.get(idx)
        if sug is None:
            continue
        sug_rel, sug_stance = sug
        auto_rel = row["auto_relevance"]
        auto_stance = row["auto_stance"]
        # 같은 값이면 빈칸 유지
        if sug_rel != auto_rel:
            row["your_relevance"] = sug_rel
        if sug_stance != auto_stance:
            row["your_stance"] = sug_stance
        if sug_rel != auto_rel or sug_stance != auto_stance:
            row["your_note"] = NOTE
            changes.append({
                "idx": idx,
                "news_id": row["news_id"],
                "pn": row["pn"],
                "alias": row["alias"],
                "from": f"rel={auto_rel}/{auto_stance}",
                "to": f"rel={sug_rel}/{sug_stance}",
            })

    with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"AI 제안 반영: {len(changes)}건 변경 (39건 중)")
    print(f"  나머지: 자동 다수결 그대로 유지\n")
    for c in changes:
        print(f"  #{c['idx']:2d} pn={c['pn']:2s} {c['alias']:20s} {c['from']:25s} → {c['to']}")
    print(f"\n다음 단계 (정석):")
    print(f"  1) open review_B_39.html  ← 39건 한 화면에 보기")
    print(f"  2) {OUTPUT}을 옆에 띄우고 each row 동의/수정")
    print(f"  3) .venv-nlp/bin/python apply_review_B_39.py  ← 최종 반영")


if __name__ == "__main__":
    main()
