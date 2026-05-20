#!/usr/bin/env python3
"""
라벨링 결과 자가 검증 스크립트
사용법:
    python validate.py train_500_labeling.csv
    python validate.py calibration_30.csv

채점하신 CSV를 발주자에게 제출하기 전에 이 스크립트로 검사해주세요.
오류가 0건이면 합격, 1건이라도 있으면 수정 후 재제출.
"""
import csv
import sys
from collections import Counter

ALLOWED_RELEVANCE = {"0", "1", "2"}
ALLOWED_STANCE = {"support", "contradict", "neutral"}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
REQUIRED_COLS = ["labeler", "relevance", "stance_label", "confidence", "reason"]


def main(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    errors = []
    rel_dist = Counter()
    stance_dist = Counter()
    conf_dist = Counter()

    for i, row in enumerate(rows, start=2):  # 헤더 다음 줄부터 2
        nid = row.get("news_id", f"row{i}")
        rel = (row.get("relevance") or "").strip()
        stance = (row.get("stance_label") or "").strip()
        conf = (row.get("confidence") or "").strip()
        reason = (row.get("reason") or "").strip()
        labeler = (row.get("labeler") or "").strip()

        if not labeler:
            errors.append(f"  [line {i}] news_id={nid}: labeler 비어있음")
        if rel not in ALLOWED_RELEVANCE:
            errors.append(f"  [line {i}] news_id={nid}: relevance '{rel}' 허용값 아님 (0/1/2)")
        if stance not in ALLOWED_STANCE:
            errors.append(f"  [line {i}] news_id={nid}: stance_label '{stance}' 허용값 아님 (support/contradict/neutral)")
        if conf not in ALLOWED_CONFIDENCE:
            errors.append(f"  [line {i}] news_id={nid}: confidence '{conf}' 허용값 아님 (low/medium/high)")
        if not reason or len(reason) < 10:
            errors.append(f"  [line {i}] news_id={nid}: reason 너무 짧거나 비어있음 ({len(reason)}자)")

        # 강제 룰: relevance=0 → stance_label=neutral
        if rel == "0" and stance != "neutral":
            errors.append(f"  [line {i}] news_id={nid}: relevance=0인데 stance_label='{stance}' (neutral 강제)")

        rel_dist[rel] += 1
        stance_dist[stance] += 1
        conf_dist[conf] += 1

    print(f"\n=== 검증 결과: {path} ===")
    print(f"총 행: {len(rows)}")
    print(f"relevance 분포: {dict(sorted(rel_dist.items()))}")
    print(f"stance_label 분포: {dict(sorted(stance_dist.items()))}")
    print(f"confidence 분포: {dict(sorted(conf_dist.items()))}")

    low_ratio = conf_dist.get("low", 0) / len(rows) if rows else 0
    print(f"\nconfidence=low 비율: {low_ratio:.1%} (권장 30% 이하)")
    if low_ratio > 0.30:
        print("  ⚠ low 비율이 높습니다. 가이드 재검토 권장.")

    print(f"\n오류: {len(errors)}건")
    if errors:
        print("\n".join(errors[:50]))
        if len(errors) > 50:
            print(f"  ... 외 {len(errors) - 50}건")
        print("\n❌ 오류를 수정 후 다시 검증해주세요.")
        sys.exit(1)
    else:
        print("\n✅ 자가 검증 통과. 발주자에게 제출 가능합니다.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python validate.py <CSV파일>")
        sys.exit(2)
    main(sys.argv[1])
