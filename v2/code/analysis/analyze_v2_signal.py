#!/usr/bin/env python3
"""
2차 PTEI 추론 시그널 분석 — 핵심 가설 검증.

분석 1: PTEI 시계열 패턴 — 정책 발표일 중심 분포
  • rel_day별 PTEI 평균 (1차 vs 2차 비교)
  • 정책×rel_day heatmap

분석 2: PTEI 방향 vs 정책 방향(+/-) 일치 검증
  • 정책별 신호 점수 = direction × (p_support - p_contradict)
  • 50개 정책 중 일치율
  • 시점별(선반영·발표·후폭풍) 일치율 비교

출력:
  - figures/v2_*.png (시각화)
  - reports/v2_signal_analysis.md (마크다운 보고서)
  - data/processed/policy_alignment_v2.csv (정책별 일치 지표)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트
rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


V2_PATH = "data/processed/article_scores_v2.csv"
V1_PATH = "data/processed/article_scores_hybrid.csv"
POLICY_CARDS = "policy_cards_v1.1.csv"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
REPORT = "reports/v2_signal_analysis.md"
ALIGNMENT_OUT = "data/processed/policy_alignment_v2.csv"


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

    print(f"[load] v2: {V2_PATH}")
    v2 = pd.read_csv(V2_PATH, encoding="utf-8-sig")
    v2["pn"] = v2["pn"].astype(int)
    v2["rel_day"] = pd.to_numeric(v2["rel_day"], errors="coerce")
    v2["direction"] = v2["pn"].map(pn_dir)
    v2["direction_sign"] = v2["direction"].map({"+": 1, "-": -1})

    print(f"[load] v1: {V1_PATH}")
    v1 = pd.read_csv(V1_PATH, encoding="utf-8-sig")
    v1["pn"] = v1["pn"].astype(int)
    v1["rel_day"] = pd.to_numeric(v1["rel_day"], errors="coerce")
    v1["direction"] = v1["pn"].map(pn_dir)
    v1["direction_sign"] = v1["direction"].map({"+": 1, "-": -1})

    # ===== 분석 1: 시계열 패턴 =====
    print("\n[analysis 1] 시계열 패턴 (D-7~D+5)")
    WIN_LO, WIN_HI = -7, 5
    v2_win = v2[(v2["rel_day"] >= WIN_LO) & (v2["rel_day"] <= WIN_HI)]
    v1_win = v1[(v1["rel_day"] >= WIN_LO) & (v1["rel_day"] <= WIN_HI)]

    daily_v2 = v2_win.groupby("rel_day").agg(
        n=("news_id", "count"),
        ptei_mean=("ptei_score", "mean"),
        p_support_mean=("p_support", "mean"),
        p_contradict_mean=("p_contradict", "mean"),
    ).reset_index()
    daily_v1 = v1_win.groupby("rel_day").agg(
        n=("news_id", "count"),
        ptei_total_mean=("PTEI_total", "mean"),
        contra_mean=("Contradiction_total", "mean"),
    ).reset_index()

    # 그림 1: 시점별 PTEI 평균 (1차 vs 2차)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(daily_v2["rel_day"], daily_v2["ptei_mean"], "o-", color="#1a73e8", lw=2, label="v2 PTEI")
    axes[0].axvline(0, color="red", ls="--", alpha=0.5, label="정책 발표일 (D0)")
    axes[0].set_xlabel("rel_day (정책 발표일 = 0)")
    axes[0].set_ylabel("PTEI 평균")
    axes[0].set_title("2차 (fine-tuned) PTEI 시점별 평균")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(daily_v1["rel_day"], daily_v1["ptei_total_mean"], "o-", color="#888", lw=2, label="v1 PTEI_total")
    axes[1].axvline(0, color="red", ls="--", alpha=0.5, label="D0")
    axes[1].set_xlabel("rel_day")
    axes[1].set_ylabel("PTEI_total 평균")
    axes[1].set_title("1차 (zero-shot) PTEI 시점별 평균")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v2_fig1_ptei_timeline.png", dpi=130)
    plt.close()
    print(f"  saved: {FIG_DIR}/v2_fig1_ptei_timeline.png")

    # 그림 2: 정책×시점 PTEI heatmap (2차)
    pivot = v2_win.pivot_table(index="pn", columns="rel_day", values="ptei_score", aggfunc="mean")
    pivot = pivot.reindex(sorted(pn_dir.keys()))
    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(p)} {pn_alias.get(int(p),'')[:12]} ({pn_dir.get(int(p),'')})" for p in pivot.index], fontsize=8)
    ax.set_xlabel("rel_day")
    ax.set_title("2차 PTEI 정책×시점 매트릭스 (열=발표일 기준 일자)")
    # D0 강조
    if 0 in pivot.columns:
        d0_idx = list(pivot.columns).index(0)
        ax.axvline(d0_idx, color="red", lw=1.5, alpha=0.6)
    plt.colorbar(im, ax=ax, label="PTEI 평균")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v2_fig2_heatmap.png", dpi=130)
    plt.close()
    print(f"  saved: {FIG_DIR}/v2_fig2_heatmap.png")

    # ===== 분석 2: 정책 방향 일치 검증 =====
    print("\n[analysis 2] 정책 방향 일치 검증")

    # 신호 점수: direction × (p_support - p_contradict)
    # direction=+ 면 support 강할수록 일치 점수 ↑
    # direction=- 면 contradict 강할수록 일치 점수 ↑
    v2_win["signal_score"] = v2_win["direction_sign"] * (v2_win["p_support"] - v2_win["p_contradict"])

    # 시점 구간 정의
    def phase(rd):
        if rd < 0:
            return "선반영 (D-7~D-1)"
        if rd <= 1:
            return "발표 (D0~D+1)"
        return "후폭풍 (D+2~D+5)"

    v2_win = v2_win.copy()
    v2_win["phase"] = v2_win["rel_day"].apply(phase)

    # 정책별 일치 지표
    by_policy = v2_win.groupby("pn").agg(
        n=("news_id", "count"),
        direction=("direction", "first"),
        signal_mean=("signal_score", "mean"),
        p_support_mean=("p_support", "mean"),
        p_contradict_mean=("p_contradict", "mean"),
    ).reset_index()
    by_policy["alias"] = by_policy["pn"].map(pn_alias)
    by_policy["aligned"] = by_policy["signal_mean"] > 0
    by_policy = by_policy[["pn", "alias", "direction", "n", "p_support_mean", "p_contradict_mean", "signal_mean", "aligned"]]
    by_policy.to_csv(ALIGNMENT_OUT, index=False, encoding="utf-8-sig")

    aligned_count = int(by_policy["aligned"].sum())
    total = len(by_policy)
    print(f"  정책 50건 중 정책 방향과 뉴스 신호가 일치: {aligned_count}/{total} ({aligned_count/total:.1%})")

    # 시점별 일치율
    phase_align = v2_win.groupby(["pn", "phase"]).agg(
        signal_mean=("signal_score", "mean"),
        direction=("direction", "first"),
    ).reset_index()
    phase_align["aligned"] = phase_align["signal_mean"] > 0
    phase_summary = phase_align.groupby("phase")["aligned"].agg(["sum", "count"])
    phase_summary["rate"] = phase_summary["sum"] / phase_summary["count"]
    print(f"\n  시점별 일치율:")
    for phase_name, row in phase_summary.iterrows():
        print(f"    {phase_name}: {int(row['sum'])}/{int(row['count'])} ({row['rate']:.1%})")

    # 그림 3: 정책별 신호 점수 (정렬, 색=방향)
    plot_df = by_policy.sort_values("signal_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ["#1a73e8" if d == "+" else "#d93025" for d in plot_df["direction"]]
    ax.barh(range(len(plot_df)), plot_df["signal_mean"], color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([f"{int(r['pn'])} {r['alias'][:18]} ({r['direction']})" for _, r in plot_df.iterrows()], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("일치 점수 (>0 이면 뉴스가 정책 방향 따라감)")
    ax.set_title("정책 50건 — 뉴스가 정책 방향과 일치하는가\n(D-7~D+5 평균, 파랑=호재정책, 빨강=악재정책)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v2_fig3_alignment_per_policy.png", dpi=130)
    plt.close()
    print(f"  saved: {FIG_DIR}/v2_fig3_alignment_per_policy.png")

    # 그림 4: 시점별 일치율
    phase_order = ["선반영 (D-7~D-1)", "발표 (D0~D+1)", "후폭풍 (D+2~D+5)"]
    phase_summary = phase_summary.reindex(phase_order)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(phase_summary)), phase_summary["rate"], color=["#34a853", "#1a73e8", "#fbbc04"])
    ax.set_xticks(range(len(phase_summary)))
    ax.set_xticklabels(phase_summary.index)
    ax.set_ylabel("정책 방향 일치율")
    ax.set_title("시점별 — 뉴스가 정책 방향을 따라가는 비율")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="50% (랜덤 수준)")
    for i, (bar, row) in enumerate(zip(bars, phase_summary.itertuples())):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{int(row.sum)}/{int(row.count)}\n({row.rate:.1%})",
                ha="center", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v2_fig4_phase_alignment.png", dpi=130)
    plt.close()
    print(f"  saved: {FIG_DIR}/v2_fig4_phase_alignment.png")

    # 그림 5: 방향(+/-) 정책별 시점 신호
    v2_win_pos = v2_win[v2_win["direction_sign"] == 1]
    v2_win_neg = v2_win[v2_win["direction_sign"] == -1]
    daily_pos = v2_win_pos.groupby("rel_day").agg(p_support_mean=("p_support", "mean"), p_contra_mean=("p_contradict", "mean")).reset_index()
    daily_neg = v2_win_neg.groupby("rel_day").agg(p_support_mean=("p_support", "mean"), p_contra_mean=("p_contradict", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(daily_pos["rel_day"], daily_pos["p_support_mean"], "o-", color="#1a73e8", label="p_support")
    axes[0].plot(daily_pos["rel_day"], daily_pos["p_contra_mean"], "o-", color="#d93025", label="p_contradict")
    axes[0].axvline(0, color="black", ls="--", alpha=0.4)
    axes[0].set_title("호재 정책(+) — 시점별 뉴스 방향")
    axes[0].set_xlabel("rel_day"); axes[0].set_ylabel("평균 확률")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(daily_neg["rel_day"], daily_neg["p_support_mean"], "o-", color="#1a73e8", label="p_support")
    axes[1].plot(daily_neg["rel_day"], daily_neg["p_contra_mean"], "o-", color="#d93025", label="p_contradict")
    axes[1].axvline(0, color="black", ls="--", alpha=0.4)
    axes[1].set_title("악재 정책(-) — 시점별 뉴스 방향")
    axes[1].set_xlabel("rel_day"); axes[1].set_ylabel("평균 확률")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v2_fig5_direction_split.png", dpi=130)
    plt.close()
    print(f"  saved: {FIG_DIR}/v2_fig5_direction_split.png")

    # ===== 마크다운 보고서 =====
    pos_policies = by_policy[by_policy["direction"] == "+"]
    neg_policies = by_policy[by_policy["direction"] == "-"]
    pos_aligned = int(pos_policies["aligned"].sum())
    neg_aligned = int(neg_policies["aligned"].sum())
    top5 = by_policy.nlargest(5, "signal_mean")[["pn", "alias", "direction", "signal_mean"]]
    bot5 = by_policy.nsmallest(5, "signal_mean")[["pn", "alias", "direction", "signal_mean"]]

    report = f"""# 2차 PTEI 신호 분석 — 핵심 가설 검증

## 본 보고서가 검증하는 것
> **신문기사의 영향력으로 정책 방향성에 맞는 주가 흐름이 보여진다** — 그 전제 조건인 "뉴스가 정책 방향과 일치하는가?"를 먼저 검증.

CAR(서강대) 데이터 도착 전, **뉴스 자체 시그널만으로** 검증 가능한 부분을 정리합니다.

---

## 핵심 결과 한눈에

### 1. 정책 50건 중 뉴스 방향 일치
- **전체 일치율**: {aligned_count}/{total} 정책 ({aligned_count/total:.1%})
- 호재 정책(+): {pos_aligned}/{len(pos_policies)} ({pos_aligned/max(1,len(pos_policies)):.1%}) 일치
- 악재 정책(-): {neg_aligned}/{len(neg_policies)} ({neg_aligned/max(1,len(neg_policies)):.1%}) 일치

### 2. 시점별 일치율
"""
    for phase_name, row in phase_summary.iterrows():
        report += f"- **{phase_name}**: {int(row['sum'])}/{int(row['count'])} 정책 ({row['rate']:.1%})\n"

    report += f"""

---

## 그림

1. [v2_fig1_ptei_timeline.png](../figures/v2_fig1_ptei_timeline.png) — 1차 vs 2차 PTEI 시점별 평균
2. [v2_fig2_heatmap.png](../figures/v2_fig2_heatmap.png) — 정책×시점 PTEI 매트릭스
3. [v2_fig3_alignment_per_policy.png](../figures/v2_fig3_alignment_per_policy.png) — 정책별 일치 점수
4. [v2_fig4_phase_alignment.png](../figures/v2_fig4_phase_alignment.png) — 시점별 일치율
5. [v2_fig5_direction_split.png](../figures/v2_fig5_direction_split.png) — 호재/악재 정책 시점별 신호

---

## 정책별 신호 (Top 5 일치 / Bottom 5)

### Top 5 — 뉴스가 정책 방향과 가장 잘 일치
| pn | alias | direction | signal_mean |
|---|---|---|---|
"""
    for _, r in top5.iterrows():
        report += f"| {int(r['pn'])} | {r['alias']} | {r['direction']} | {r['signal_mean']:+.3f} |\n"

    report += "\n### Bottom 5 — 뉴스가 정책 방향과 가장 어긋남\n"
    report += "| pn | alias | direction | signal_mean |\n|---|---|---|---|\n"
    for _, r in bot5.iterrows():
        report += f"| {int(r['pn'])} | {r['alias']} | {r['direction']} | {r['signal_mean']:+.3f} |\n"

    report += f"""

---

## 해석 가이드

- **signal_mean > 0**: 뉴스가 정책 의도 방향을 그대로 보도 (호재 정책엔 호재 톤, 악재 정책엔 악재 톤)
- **signal_mean < 0**: 뉴스가 정책 의도와 반대로 보도 (호재 정책에 비판·우려 톤 등)
- **시점별 차이**: 선반영 단계에서 일치율이 높으면 = 뉴스가 정책 발표를 미리 예고
- **D0~D+1 일치율**: 가장 강해야 정상 (발표 직후 공식 보도)

---

## 데이터 출처

- 입력: `data/processed/article_scores_v2.csv` (fine-tuned KLUE-RoBERTa, 99,539건)
- 정책 방향: `policy_cards_v1.1.csv` (D_i_initial)
- 윈도우: rel_day ∈ [{WIN_LO}, {WIN_HI}]

## 다음 단계

서강대 CAR(누적초과수익률) 도착 시:
1. PTEI peak day vs CAR peak day 비교
2. PTEI × CAR cross-correlation (lag 분석)
3. 4유형 분류 (즉시·선반영·지연·설명보강)
"""

    Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[done]")
    print(f"  report: {REPORT}")
    print(f"  alignment csv: {ALIGNMENT_OUT}")
    print(f"\n핵심 결과:")
    print(f"  ✅ 정책 50건 중 {aligned_count}/{total} ({aligned_count/total:.1%}) 정책에서 뉴스가 정책 방향과 일치")
    print(f"  ✅ 호재 정책 일치율: {pos_aligned}/{len(pos_policies)} ({pos_aligned/max(1,len(pos_policies)):.1%})")
    print(f"  ✅ 악재 정책 일치율: {neg_aligned}/{len(neg_policies)} ({neg_aligned/max(1,len(neg_policies)):.1%})")


if __name__ == "__main__":
    main()
