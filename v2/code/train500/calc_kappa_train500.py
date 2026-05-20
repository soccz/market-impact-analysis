#!/usr/bin/env python3
"""
Train 500 라벨링 결과 일치도 측정.
- pairwise Cohen's κ (1-2, 1-3, 2-3) on relevance / stance_label
- Fleiss' κ (3 raters) on relevance / stance_label
- relevance × stance 결합 라벨에 대한 κ도 보조 산출
- 분포·합의 패턴 요약

statsmodels 없이 sklearn만 사용 (Cohen's κ) + Fleiss' κ 직접 구현.
"""
import csv
from collections import Counter, defaultdict
from sklearn.metrics import cohen_kappa_score


LABELERS = ["1", "2", "3"]
INPUT_TPL = "외주/train_500_{}_final.csv"
KEY = "news_id"


def load(labeler):
    path = INPUT_TPL.format(labeler)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r[KEY]: r for r in csv.DictReader(f)}


def fleiss_kappa(ratings, categories):
    """
    ratings: list of dicts, each dict {category: count} summing to N raters.
    categories: list of category labels.
    Returns Fleiss' kappa.
    """
    N = len(ratings)
    if N == 0:
        return float("nan")
    n_raters = sum(ratings[0].values())
    # P_i: agreement per item
    P_is = []
    for r in ratings:
        s = sum(c * (c - 1) for c in r.values())
        P_is.append(s / (n_raters * (n_raters - 1)))
    P_bar = sum(P_is) / N
    # p_j: proportion of all assignments to category j
    totals = Counter()
    for r in ratings:
        for c, v in r.items():
            totals[c] += v
    P_e = sum((totals[c] / (N * n_raters)) ** 2 for c in categories)
    if 1 - P_e == 0:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def to_fleiss_rows(label_lists, categories):
    """label_lists: list of lists (n_items × n_raters) → list of category-count dicts."""
    rows = []
    for item_labels in label_lists:
        d = {c: 0 for c in categories}
        for lab in item_labels:
            d[lab] += 1
        rows.append(d)
    return rows


def main():
    data = {l: load(l) for l in LABELERS}
    common_ids = set.intersection(*(set(d.keys()) for d in data.values()))
    print(f"공통 news_id: {len(common_ids)}건 (각자 500건 기준)\n")

    common_ids = sorted(common_ids)
    rel = {l: [data[l][i]["relevance"] for i in common_ids] for l in LABELERS}
    stance = {l: [data[l][i]["stance_label"] for i in common_ids] for l in LABELERS}

    # rel × stance 결합 라벨 (총 9가지)
    combo = {l: [f"{rel[l][k]}_{stance[l][k]}" for k in range(len(common_ids))] for l in LABELERS}

    print("=" * 60)
    print("[A] Pairwise Cohen's κ")
    print("=" * 60)
    for a, b in [("1", "2"), ("1", "3"), ("2", "3")]:
        k_rel = cohen_kappa_score(rel[a], rel[b])
        k_stance = cohen_kappa_score(stance[a], stance[b])
        k_combo = cohen_kappa_score(combo[a], combo[b])
        print(f"  {a} ↔ {b}:  relevance κ = {k_rel:.3f}  |  stance κ = {k_stance:.3f}  |  combo κ = {k_combo:.3f}")

    print()
    print("=" * 60)
    print("[B] Fleiss' κ (3 raters)")
    print("=" * 60)
    rel_cats = ["0", "1", "2"]
    stance_cats = ["support", "contradict", "neutral"]
    rel_rows = to_fleiss_rows(list(zip(rel["1"], rel["2"], rel["3"])), rel_cats)
    stance_rows = to_fleiss_rows(list(zip(stance["1"], stance["2"], stance["3"])), stance_cats)
    print(f"  relevance Fleiss κ = {fleiss_kappa(rel_rows, rel_cats):.3f}")
    print(f"  stance    Fleiss κ = {fleiss_kappa(stance_rows, stance_cats):.3f}")

    print()
    print("=" * 60)
    print("[C] 합의 패턴 (3명 동시)")
    print("=" * 60)
    patterns_rel = Counter()
    patterns_stance = Counter()
    for k in range(len(common_ids)):
        rel_vals = sorted([rel["1"][k], rel["2"][k], rel["3"][k]])
        stance_vals = sorted([stance["1"][k], stance["2"][k], stance["3"][k]])

        if rel_vals[0] == rel_vals[2]:
            patterns_rel["3:0 (만장일치)"] += 1
        elif rel_vals[0] == rel_vals[1] or rel_vals[1] == rel_vals[2]:
            patterns_rel["2:1 (다수결)"] += 1
        else:
            patterns_rel["1:1:1 (3 다름)"] += 1

        if stance_vals[0] == stance_vals[2]:
            patterns_stance["3:0 (만장일치)"] += 1
        elif stance_vals[0] == stance_vals[1] or stance_vals[1] == stance_vals[2]:
            patterns_stance["2:1 (다수결)"] += 1
        else:
            patterns_stance["1:1:1 (3 다름)"] += 1

    print(f"  relevance: {dict(patterns_rel)}")
    print(f"  stance:    {dict(patterns_stance)}")

    # 학술 해석 가이드
    print()
    print("=" * 60)
    print("[D] 학술 해석 가이드 (Landis & Koch 1977)")
    print("=" * 60)
    print("  κ <0.00 poor / 0.00-0.20 slight / 0.21-0.40 fair / 0.41-0.60 moderate")
    print("  0.61-0.80 substantial / 0.81-1.00 almost perfect")


if __name__ == "__main__":
    main()
