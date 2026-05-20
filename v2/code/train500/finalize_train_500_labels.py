#!/usr/bin/env python3
"""
Train 500 합의 라벨 최종본 생성 (자동 다수결, 본인 중재 없음).

방법:
- 만장일치: 그대로 채택
- 2:1 다수결: confidence 가중치 없이 단순 다수결 채택
- 결과 컬럼: relevance, stance_label (기존 컬럼 그대로 사용)
- 추가 메타: consensus_status, agreement_score, labeler="consensus_majority_vote"

학술 메서드 표기 (보고서·논문용):
  "Three independent human annotators labeled 500 articles using the same
   labeling guide (v1.1). Pairwise Cohen's κ ranged 0.55-0.83 (relevance)
   and 0.75-0.89 (stance). Fleiss' κ = 0.698 (relevance), 0.818 (stance).
   All disagreements were resolved by majority voting. Low-confidence
   ratings were not down-weighted."
"""
import csv
from collections import Counter


INPUT = "splits/train_500_consensus.csv"
OUTPUT = "splits/train_500_final.csv"
METHOD = "splits/train_500_method.md"


def main():
    with open(INPUT, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # 기존 train_500_labeling.csv 컬럼 구조 + 메타 추가
    base_cols = ["pn", "alias", "direction", "industry", "rel_day", "pub_date",
                 "provider", "title", "content", "news_id", "category"]
    out_cols = base_cols + [
        "labeler", "relevance", "stance_label", "confidence", "reason",
        "consensus_status", "agreement_score",
        "rel_1", "rel_2", "rel_3", "stance_1", "stance_2", "stance_3",
        "conf_1", "conf_2", "conf_3",
    ]

    out_rows = []
    status_count = Counter()
    for r in rows:
        rels = [r["rel_1"], r["rel_2"], r["rel_3"]]
        stances = [r["stance_1"], r["stance_2"], r["stance_3"]]
        confs = [r["conf_1"], r["conf_2"], r["conf_3"]]

        # 단순 다수결 (동수면 가장 빈도 높은 첫 번째)
        rel_final = Counter(rels).most_common(1)[0][0]
        stance_final = Counter(stances).most_common(1)[0][0]
        rel_final_str = str(rel_final)
        # rel=0 강제 룰: stance=neutral
        if rel_final_str == "0":
            stance_final = "neutral"

        # confidence는 다수결 (low/medium/high 중 가장 빈도 높음)
        conf_final = Counter(confs).most_common(1)[0][0]

        # 합의 점수: 3명 중 일치율
        rel_agreement = max(Counter(rels).values()) / 3
        stance_agreement = max(Counter(stances).values()) / 3
        agreement = round((rel_agreement + stance_agreement) / 2, 3)

        status = r["consensus_status"]
        status_count[status] += 1

        reason = f"[majority of 3 annotators] rel={rel_final}/stance={stance_final} "
        reason += f"(rel votes: {dict(Counter(rels))}, stance votes: {dict(Counter(stances))})"

        out = {c: r.get(c, "") for c in base_cols}
        out.update({
            "labeler": "consensus_majority_vote",
            "relevance": rel_final_str,
            "stance_label": stance_final,
            "confidence": conf_final,
            "reason": reason,
            "consensus_status": status,
            "agreement_score": agreement,
            "rel_1": rels[0], "rel_2": rels[1], "rel_3": rels[2],
            "stance_1": stances[0], "stance_2": stances[1], "stance_3": stances[2],
            "conf_1": confs[0], "conf_2": confs[1], "conf_3": confs[2],
        })
        out_rows.append(out)

    with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out_rows)

    # 분포 요약
    final_rel = Counter(r["relevance"] for r in out_rows)
    final_stance = Counter(r["stance_label"] for r in out_rows)
    final_conf = Counter(r["confidence"] for r in out_rows)
    agreement_dist = Counter(r["agreement_score"] for r in out_rows)

    # 메서드 문서
    method = f"""# Train 500 합의 라벨 — 메서드

## 데이터 출처
- 입력: 3명의 독립 라벨러(annotator 1/2/3)가 동일한 가이드 v1.1으로 채점한 500건
- 라벨 컬럼: `relevance` (0/1/2), `stance_label` (support/contradict/neutral), `confidence` (low/medium/high)

## 일치도 (학술 보고용)
- **Pairwise Cohen's κ**
  - 1↔2: relevance 0.825, stance 0.887
  - 1↔3: relevance 0.552, stance 0.751
  - 2↔3: relevance 0.715, stance 0.817
- **Fleiss' κ (3 raters)**
  - relevance **0.698** (substantial)
  - stance **0.818** (almost perfect)

## 합의 도출
- **만장일치 (3:0)**: 333건 (66.6%) — 그대로 채택
- **다수결 (2:1) 자동 채택**: 167건 (33.4%, 만장일치 아닌 전체)
- 본인 중재 없음, low confidence 가중치 없음

## 강제 룰
- `relevance = 0` → `stance_label = neutral`

## agreement_score
- 각 행에 (relevance 일치율 + stance 일치율) / 2 기록
- 1.0 = 만장일치, 0.667 = 2:1, 0.5 = 1:1:1 (실제 1:1:1은 0건)

## 분포
- relevance: {dict(sorted(final_rel.items()))}
- stance: {dict(sorted(final_stance.items()))}
- confidence: {dict(sorted(final_conf.items()))}
- agreement_score: {dict(sorted(agreement_dist.items(), reverse=True))}

## 학술 보고 권장 문장
> Three independent human annotators labeled 500 articles using the same
> labeling guide. Pairwise Cohen's κ ranged 0.55-0.83 (relevance) and
> 0.75-0.89 (stance). Fleiss' κ was 0.698 (relevance, substantial) and
> 0.818 (stance, almost perfect). All disagreements were resolved by
> majority voting; low-confidence ratings were not down-weighted.
"""
    with open(METHOD, "w", encoding="utf-8") as f:
        f.write(method)

    print(f"saved: {OUTPUT} ({len(out_rows)}건)")
    print(f"saved: {METHOD}")
    print(f"\n분포 — relevance: {dict(sorted(final_rel.items()))}")
    print(f"분포 — stance: {dict(sorted(final_stance.items()))}")
    print(f"분포 — confidence: {dict(sorted(final_conf.items()))}")
    print(f"\nlabeler 표기: 'consensus_majority_vote'")
    print(f"학술 카드: Fleiss' κ relevance 0.698, stance 0.818")


if __name__ == "__main__":
    main()
