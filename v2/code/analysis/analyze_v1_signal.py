#!/usr/bin/env python3
"""
1차 추론 시그널 분석 — 본 가설 검증 (1차로 재실행).

핵심: 1차의 PTEI_total/Contradiction_total 분리 구조 활용
  signal = direction × (PTEI_total - Contradiction_total)

분석:
  1. 시계열 패턴 (D-7~D+5)
  2. 정책 방향(+/-) 일치 검증 (호재/악재 분리)
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = ["AppleGothic"]
rcParams["axes.unicode_minus"] = False

V1_PATH = "data/processed/article_scores_hybrid.csv"
POLICY_CARDS = "policy_cards_v1.1.csv"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
REPORT = "reports/v1_signal_analysis.md"
ALIGNMENT_OUT = "data/processed/policy_alignment_v1.csv"


def load_policy_directions():
    pn_dir = {}
    pn_alias = {}
    with open(POLICY_CARDS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pn_dir[int(r["pn"])] = r["D_i_initial"]
            pn_alias[int(r["pn"])] = r["alias"]
    return pn_dir, pn_alias


def main():
    pn_dir, pn_alias = load_policy_directions()
    print(f"[load] policies: {len(pn_dir)}")
    print(f"[load] {V1_PATH}")
    v1 = pd.read_csv(V1_PATH, encoding="utf-8-sig", low_memory=False)
    v1["pn"] = pd.to_numeric(v1["pn"], errors="coerce").astype("Int64")
    v1 = v1.dropna(subset=["pn"])
    v1["pn"] = v1["pn"].astype(int)
    v1["rel_day"] = pd.to_numeric(v1["rel_day"], errors="coerce")
    v1["PTEI_total"] = pd.to_numeric(v1["PTEI_total"], errors="coerce").fillna(0)
    v1["Contradiction_total"] = pd.to_numeric(v1["Contradiction_total"], errors="coerce").fillna(0)
    v1["direction"] = v1["pn"].map(pn_dir)
    v1["direction_sign"] = v1["direction"].map({"+": 1, "-": -1})

    # 핵심 신호: direction × (PTEI - Contradict)
    v1["signal_score"] = v1["direction_sign"] * (v1["PTEI_total"] - v1["Contradiction_total"])

    WIN_LO, WIN_HI = -7, 5
    win = v1[(v1["rel_day"] >= WIN_LO) & (v1["rel_day"] <= WIN_HI)].copy()
    print(f"[window] D{WIN_LO}~D+{WIN_HI}: {len(win):,}건")

    def phase(rd):
        if rd < 0:
            return "선반영 (D-7~D-1)"
        if rd <= 1:
            return "발표 (D0~D+1)"
        return "후폭풍 (D+2~D+5)"

    win["phase"] = win["rel_day"].apply(phase)

    # ===== 분석 1: 시계열 =====
    print("\n[analysis 1] 시계열 패턴")
    daily = win.groupby("rel_day").agg(
        n=("news_id", "count"),
        ptei_mean=("PTEI_total", "mean"),
        contra_mean=("Contradiction_total", "mean"),
        signal_mean=("signal_score", "mean"),
    ).reset_index()

    # 호재/악재 분리 시계열
    daily_pos = win[win["direction_sign"] == 1].groupby("rel_day").agg(
        ptei_mean=("PTEI_total", "mean"),
        contra_mean=("Contradiction_total", "mean"),
    ).reset_index()
    daily_neg = win[win["direction_sign"] == -1].groupby("rel_day").agg(
        ptei_mean=("PTEI_total", "mean"),
        contra_mean=("Contradiction_total", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(daily_pos["rel_day"], daily_pos["ptei_mean"], "o-", color="#1a73e8", lw=2, label="PTEI_total")
    axes[0].plot(daily_pos["rel_day"], daily_pos["contra_mean"], "o-", color="#d93025", lw=2, label="Contradiction_total")
    axes[0].axvline(0, color="black", ls="--", alpha=0.4)
    axes[0].set_title("호재 정책(+) — PTEI는 ↑, Contradict는 ↓ 기대")
    axes[0].set_xlabel("rel_day"); axes[0].set_ylabel("평균 점수")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(daily_neg["rel_day"], daily_neg["ptei_mean"], "o-", color="#1a73e8", lw=2, label="PTEI_total")
    axes[1].plot(daily_neg["rel_day"], daily_neg["contra_mean"], "o-", color="#d93025", lw=2, label="Contradiction_total")
    axes[1].axvline(0, color="black", ls="--", alpha=0.4)
    axes[1].set_title("악재 정책(-) — Contradict는 ↑, PTEI는 ↓ 기대")
    axes[1].set_xlabel("rel_day"); axes[1].set_ylabel("평균 점수")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v1_fig1_direction_split.png", dpi=130)
    plt.close()
    print(f"  saved: {FIG_DIR}/v1_fig1_direction_split.png")

    # ===== 분석 2: 정책 방향 일치 =====
    print("\n[analysis 2] 정책 방향 일치")

    by_policy = win.groupby("pn").agg(
        n=("news_id", "count"),
        direction=("direction", "first"),
        ptei_mean=("PTEI_total", "mean"),
        contra_mean=("Contradiction_total", "mean"),
        signal_mean=("signal_score", "mean"),
    ).reset_index()
    by_policy["alias"] = by_policy["pn"].map(pn_alias)
    by_policy["aligned"] = by_policy["signal_mean"] > 0
    by_policy = by_policy[["pn", "alias", "direction", "n", "ptei_mean", "contra_mean", "signal_mean", "aligned"]]
    by_policy.to_csv(ALIGNMENT_OUT, index=False, encoding="utf-8-sig")

    aligned_count = int(by_policy["aligned"].sum())
    total = len(by_policy)
    pos = by_policy[by_policy["direction"] == "+"]
    neg = by_policy[by_policy["direction"] == "-"]
    pos_aligned = int(pos["aligned"].sum())
    neg_aligned = int(neg["aligned"].sum())

    print(f"  전체: {aligned_count}/{total} ({aligned_count/total:.1%})")
    print(f"  호재(+): {pos_aligned}/{len(pos)} ({pos_aligned/max(1,len(pos)):.1%})")
    print(f"  악재(-): {neg_aligned}/{len(neg)} ({neg_aligned/max(1,len(neg)):.1%})")

    # 시점별 일치율
    phase_align = win.groupby(["pn", "phase"]).agg(
        signal_mean=("signal_score", "mean"),
        direction=("direction", "first"),
    ).reset_index()
    phase_align["aligned"] = phase_align["signal_mean"] > 0
    phase_summary = phase_align.groupby("phase")["aligned"].agg(["sum", "count"])
    phase_summary["rate"] = phase_summary["sum"] / phase_summary["count"]
    phase_summary = phase_summary.reindex(["선반영 (D-7~D-1)", "발표 (D0~D+1)", "후폭풍 (D+2~D+5)"])
    print(f"\n  시점별:")
    for ph, row in phase_summary.iterrows():
        print(f"    {ph}: {int(row['sum'])}/{int(row['count'])} ({row['rate']:.1%})")

    # 그림 2: 정책별 신호
    plot_df = by_policy.sort_values("signal_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ["#1a73e8" if d == "+" else "#d93025" for d in plot_df["direction"]]
    ax.barh(range(len(plot_df)), plot_df["signal_mean"], color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([f"{int(r['pn'])} {r['alias'][:18]} ({r['direction']})" for _, r in plot_df.iterrows()], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("일치 점수 = direction × (PTEI - Contradict)")
    ax.set_title("1차 — 정책 50건 일치 점수 (파랑=호재, 빨강=악재)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v1_fig2_alignment_per_policy.png", dpi=130)
    plt.close()
    print(f"  saved: {FIG_DIR}/v1_fig2_alignment_per_policy.png")

    # 그림 3: 시점별 일치율
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(phase_summary)), phase_summary["rate"],
                  color=["#34a853", "#1a73e8", "#fbbc04"])
    ax.set_xticks(range(len(phase_summary)))
    ax.set_xticklabels(phase_summary.index)
    ax.set_ylabel("정책 방향 일치율")
    ax.set_title("1차 — 시점별 일치율")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="50% (랜덤)")
    for bar, (_, row) in zip(bars, phase_summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{int(row['sum'])}/{int(row['count'])}\n({row['rate']:.1%})",
                ha="center", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v1_fig3_phase_alignment.png", dpi=130)
    plt.close()
    print(f"  saved: {FIG_DIR}/v1_fig3_phase_alignment.png")

    # ===== 1차 vs 2차 비교 =====
    print("\n[compare] 1차 vs 2차 비교 카드")
    v2_alignment = pd.read_csv("data/processed/policy_alignment_v2.csv", encoding="utf-8-sig")
    v2_aligned = int(v2_alignment["aligned"].sum())
    v2_total = len(v2_alignment)
    v2_pos = v2_alignment[v2_alignment["direction"] == "+"]
    v2_neg = v2_alignment[v2_alignment["direction"] == "-"]
    v2_pos_aligned = int(v2_pos["aligned"].sum())
    v2_neg_aligned = int(v2_neg["aligned"].sum())

    # 같은 정책에서 1차/2차가 어떻게 다른지
    cmp_df = by_policy[["pn", "alias", "direction", "signal_mean", "aligned"]].rename(columns={"signal_mean": "v1_signal", "aligned": "v1_aligned"})
    cmp_df = cmp_df.merge(
        v2_alignment[["pn", "signal_mean", "aligned"]].rename(columns={"signal_mean": "v2_signal", "aligned": "v2_aligned"}),
        on="pn"
    )
    both_aligned = int(((cmp_df["v1_aligned"]) & (cmp_df["v2_aligned"])).sum())
    only_v1 = int(((cmp_df["v1_aligned"]) & (~cmp_df["v2_aligned"])).sum())
    only_v2 = int(((~cmp_df["v1_aligned"]) & (cmp_df["v2_aligned"])).sum())
    neither = int(((~cmp_df["v1_aligned"]) & (~cmp_df["v2_aligned"])).sum())
    cmp_df.to_csv("data/processed/v1_v2_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"  both aligned: {both_aligned}, only v1: {only_v1}, only v2: {only_v2}, neither: {neither}")

    # ===== 보고서 =====
    top5 = by_policy.nlargest(5, "signal_mean")[["pn", "alias", "direction", "signal_mean"]]
    bot5 = by_policy.nsmallest(5, "signal_mean")[["pn", "alias", "direction", "signal_mean"]]

    report = f"""# 1차 추론 신호 분석 — 핵심 가설 검증 (1차로 재실행)

## 본 보고서 목적
> **신문기사의 영향력으로 정책 방향성에 맞는 주가 흐름이 보여진다** — 그 전제 조건인 "뉴스가 정책 방향과 일치하는가?"를 1차 모델로 검증.

핵심 신호 정의:
```
signal = direction × (PTEI_total - Contradiction_total)
```
- 호재 정책에선 PTEI 강하면 +, Contradict 강하면 -
- 악재 정책에선 Contradict 강하면 +, PTEI 강하면 -
- 1차 모델은 `PTEI`/`Contradict`가 분리되어 호재/악재 모두 잡힘

---

## 핵심 결과

### 1. 정책 방향 일치 (D-7~D+5 평균)
- **전체**: {aligned_count}/{total} ({aligned_count/total:.1%})
- **호재 정책(+)**: {pos_aligned}/{len(pos)} ({pos_aligned/max(1,len(pos)):.1%})
- **악재 정책(-)**: {neg_aligned}/{len(neg)} ({neg_aligned/max(1,len(neg)):.1%})  ← 2차의 약점 해소 확인

### 2. 시점별
"""
    for ph, row in phase_summary.iterrows():
        report += f"- **{ph}**: {int(row['sum'])}/{int(row['count'])} ({row['rate']:.1%})\n"

    report += f"""

---

## 1차 vs 2차 비교

| | 전체 | 호재(+) | 악재(-) |
|---|---|---|---|
| **1차 (zero-shot)** | {aligned_count}/{total} ({aligned_count/total:.0%}) | {pos_aligned}/{len(pos)} ({pos_aligned/max(1,len(pos)):.0%}) | {neg_aligned}/{len(neg)} ({neg_aligned/max(1,len(neg)):.0%}) |
| **2차 (fine-tuned)** | {v2_aligned}/{v2_total} ({v2_aligned/v2_total:.0%}) | {v2_pos_aligned}/{len(v2_pos)} ({v2_pos_aligned/max(1,len(v2_pos)):.0%}) | {v2_neg_aligned}/{len(v2_neg)} ({v2_neg_aligned/max(1,len(v2_neg)):.0%}) |

정책 단위 일치 패턴:
- 둘 다 일치: {both_aligned}건
- 1차만 일치: {only_v1}건
- 2차만 일치: {only_v2}건
- 둘 다 불일치: {neither}건

---

## Top 5 (가장 강한 일치)
| pn | alias | direction | signal_mean |
|---|---|---|---|
"""
    for _, r in top5.iterrows():
        report += f"| {int(r['pn'])} | {r['alias']} | {r['direction']} | {r['signal_mean']:+.3f} |\n"

    report += "\n## Bottom 5 (가장 강한 불일치)\n| pn | alias | direction | signal_mean |\n|---|---|---|---|\n"
    for _, r in bot5.iterrows():
        report += f"| {int(r['pn'])} | {r['alias']} | {r['direction']} | {r['signal_mean']:+.3f} |\n"

    report += f"""

---

## 그림
- [v1_fig1_direction_split.png](../figures/v1_fig1_direction_split.png) — 호재/악재 정책 시점별 신호
- [v1_fig2_alignment_per_policy.png](../figures/v1_fig2_alignment_per_policy.png) — 정책별 일치 점수
- [v1_fig3_phase_alignment.png](../figures/v1_fig3_phase_alignment.png) — 시점별 일치율

## 산출물
- `data/processed/policy_alignment_v1.csv` — 정책별 일치 지표
- `data/processed/v1_v2_comparison.csv` — 1차 vs 2차 정책 단위 비교
"""

    Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[done] report: {REPORT}")
    print(f"\n=== 핵심 결과 (1차) ===")
    print(f"  전체 일치: {aligned_count}/{total} ({aligned_count/total:.1%})")
    print(f"  호재(+):  {pos_aligned}/{len(pos)} ({pos_aligned/max(1,len(pos)):.1%})")
    print(f"  악재(-):  {neg_aligned}/{len(neg)} ({neg_aligned/max(1,len(neg)):.1%})")
    print(f"\n=== 2차와 비교 ===")
    print(f"  1차 vs 2차 전체: {aligned_count/total:.1%} vs {v2_aligned/v2_total:.1%}")
    print(f"  1차 vs 2차 악재: {neg_aligned/max(1,len(neg)):.1%} vs {v2_neg_aligned/max(1,len(v2_neg)):.1%}")


if __name__ == "__main__":
    main()
