#!/usr/bin/env python3
"""
Train 500 라벨링 3명 결과로 합의 라벨 도출 + 본인 중재 큐 생성.

룰:
- 3:0 만장일치 → 자동 채택 (consensus_status = "unanimous")
- 2:1 + 3명 모두 high/medium → 다수결 자동 채택 (consensus_status = "majority_auto")
- 2:1 + 누구든 confidence=low → 본인 중재 큐 (consensus_status = "needs_review_low")
- 1:1:1 (3명 다름) → 본인 중재 큐 (consensus_status = "needs_review_split")

산출:
- splits/train_500_consensus.csv     : 합의된 라벨 (중재 전)
- splits/train_500_review_queue.csv  : 본인이 봐야 할 행만 추출
"""
import csv
from collections import Counter
from collections import defaultdict


LABELERS = ["1", "2", "3"]
INPUT_TPL = "외주/train_500_{}_final.csv"
OUT_CONSENSUS = "splits/train_500_consensus.csv"
OUT_REVIEW = "splits/train_500_review_queue.csv"


def majority(values):
    """다수결. 동수면 None."""
    c = Counter(values)
    top = c.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def main():
    data = {l: {} for l in LABELERS}
    for l in LABELERS:
        with open(INPUT_TPL.format(l), encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                data[l][row["news_id"]] = row

    common = sorted(set.intersection(*(set(d.keys()) for d in data.values())))
    base_cols = ["pn", "alias", "direction", "industry", "rel_day", "pub_date",
                 "provider", "title", "content", "news_id", "category"]
    out_cols = base_cols + [
        "consensus_relevance", "consensus_stance", "consensus_status",
        "rel_1", "rel_2", "rel_3",
        "stance_1", "stance_2", "stance_3",
        "conf_1", "conf_2", "conf_3",
        "reason_1", "reason_2", "reason_3",
    ]

    consensus_rows = []
    review_rows = []
    status_counter = Counter()

    for nid in common:
        rows = {l: data[l][nid] for l in LABELERS}
        rels = [rows[l]["relevance"] for l in LABELERS]
        stances = [rows[l]["stance_label"] for l in LABELERS]
        confs = [rows[l]["confidence"] for l in LABELERS]

        rel_consensus = majority(rels)
        stance_consensus = majority(stances)

        rel_unanimous = len(set(rels)) == 1
        stance_unanimous = len(set(stances)) == 1
        any_low = "low" in confs

        if rel_unanimous and stance_unanimous:
            status = "unanimous"
        elif rel_consensus is not None and stance_consensus is not None:
            if any_low:
                status = "needs_review_low"
            else:
                status = "majority_auto"
        else:
            # 한 차원이라도 1:1:1 splits
            status = "needs_review_split"

        # consensus 값: 자동 채택 또는 빈칸(중재 필요)
        if status in ("unanimous", "majority_auto"):
            cons_rel = rel_consensus
            cons_stance = stance_consensus
        else:
            cons_rel = rel_consensus if rel_consensus is not None else ""
            cons_stance = stance_consensus if stance_consensus is not None else ""

        out = {c: rows["1"].get(c, "") for c in base_cols}
        out["consensus_relevance"] = cons_rel
        out["consensus_stance"] = cons_stance
        out["consensus_status"] = status
        for i, l in enumerate(LABELERS, start=1):
            out[f"rel_{i}"] = rows[l]["relevance"]
            out[f"stance_{i}"] = rows[l]["stance_label"]
            out[f"conf_{i}"] = rows[l]["confidence"]
            out[f"reason_{i}"] = rows[l]["reason"]
        consensus_rows.append(out)
        if status.startswith("needs_review"):
            review_rows.append(out)
        status_counter[status] += 1

    with open(OUT_CONSENSUS, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(consensus_rows)

    with open(OUT_REVIEW, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(review_rows)

    print(f"saved: {OUT_CONSENSUS} ({len(consensus_rows)}건)")
    print(f"saved: {OUT_REVIEW} ({len(review_rows)}건)")
    print(f"\n분류 결과:")
    total = sum(status_counter.values())
    for s, n in sorted(status_counter.items(), key=lambda x: -x[1]):
        print(f"  {s:30s}: {n:4d}건 ({n/total:.1%})")
    print(f"\n본인 중재 필요: {len(review_rows)}건 (예상 1~3시간)")


if __name__ == "__main__":
    main()
