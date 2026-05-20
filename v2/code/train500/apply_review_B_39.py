#!/usr/bin/env python3
"""
splits/review_B_39_input.csv에 적은 본인 결정을 train_500_final.csv에 반영.

룰:
- your_relevance / your_stance 가 빈칸이면 → 자동 다수결 유지 (변경 없음)
- 둘 다 채워져 있으면 → 본인 결정으로 덮어쓰기
- 하나만 채워져 있으면 → 그 컬럼만 덮어쓰기 (다른 건 자동 유지)
- relevance=0 → stance=neutral 강제 룰 재적용
- labeler 표기를 "consensus_majority_vote + expert_review" 로 변경 (덮어쓴 행만)
- reason에 본인 결정 사유 추가
"""
import csv
import shutil
from collections import Counter


INPUT_REVIEW = "review_B_39_input.csv"
INPUT_FINAL = "splits/train_500_final.csv"
OUTPUT_FINAL = "splits/train_500_final.csv"
BACKUP_FINAL = "splits/train_500_final.before_review.csv"


def main():
    # 백업
    shutil.copy(INPUT_FINAL, BACKUP_FINAL)
    print(f"백업: {INPUT_FINAL} → {BACKUP_FINAL}")

    # 본인 결정 로드
    reviews = {}
    with open(INPUT_REVIEW, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            yr = (r.get("your_relevance") or "").strip()
            ys = (r.get("your_stance") or "").strip()
            yn = (r.get("your_note") or "").strip()
            if yr or ys or yn:
                reviews[r["news_id"]] = {
                    "your_relevance": yr,
                    "your_stance": ys,
                    "your_note": yn,
                }

    print(f"본인 결정 적용 대상: {len(reviews)}건")
    if len(reviews) == 0:
        print("→ 변경 없음. 자동 다수결 유지.")
        return

    # final.csv 업데이트
    with open(INPUT_FINAL, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    changed = 0
    for r in rows:
        nid = r["news_id"]
        if nid not in reviews:
            continue
        rv = reviews[nid]
        old_rel = r["relevance"]
        old_stance = r["stance_label"]
        new_rel = rv["your_relevance"] or old_rel
        new_stance = rv["your_stance"] or old_stance
        # 룰: rel=0 → stance=neutral
        if str(new_rel) == "0":
            new_stance = "neutral"
        if new_rel == old_rel and new_stance == old_stance and not rv["your_note"]:
            continue
        r["relevance"] = str(new_rel)
        r["stance_label"] = new_stance
        r["labeler"] = "consensus_majority_vote + expert_review (author)"
        note = f" [expert review: rel {old_rel}→{new_rel}, stance {old_stance}→{new_stance}"
        if rv["your_note"]:
            note += f"; note: {rv['your_note']}"
        note += "]"
        r["reason"] = r["reason"] + note
        changed += 1

    with open(OUTPUT_FINAL, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"적용 완료: {changed}건 변경")
    print(f"산출: {OUTPUT_FINAL}")
    print(f"롤백 필요 시: cp {BACKUP_FINAL} {OUTPUT_FINAL}")

    # 변경 후 분포
    rel_dist = Counter(r["relevance"] for r in rows)
    stance_dist = Counter(r["stance_label"] for r in rows)
    print(f"\n변경 후 분포:")
    print(f"  relevance: {dict(sorted(rel_dist.items()))}")
    print(f"  stance: {dict(sorted(stance_dist.items()))}")


if __name__ == "__main__":
    main()
