#!/usr/bin/env python3
"""악재 17건 PTEI vs Contradict scatter — Part 9.8용."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = ["AppleGothic"]
rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/processed/negative_policy_audit.csv", encoding="utf-8-sig")

# 카테고리
def cat(r):
    if r["aligned"]:
        return "A. 정확히 잡음"
    if r["suspicious_d_i"]:
        return "C. D_i 의심"
    return "B. 양면 보도"

df["category"] = df.apply(cat, axis=1)
colors = {"A. 정확히 잡음": "#2d9a6a", "B. 양면 보도": "#c9961f", "C. D_i 의심": "#c44a64"}

fig, ax = plt.subplots(figsize=(10, 7))

# 대각선 (PTEI = Contra)
max_v = max(df["ptei_mean"].max(), df["contra_mean"].max()) * 1.1
ax.plot([0, max_v], [0, max_v], "--", color="#888", alpha=0.5, lw=1, label="PTEI = Contradict (균형선)")

for c, color in colors.items():
    sub = df[df["category"] == c]
    ax.scatter(sub["ptei_mean"], sub["contra_mean"], c=color, s=120, alpha=0.75, edgecolors="white", lw=1.5, label=c)
    for _, r in sub.iterrows():
        ax.annotate(f"  {int(r['pn'])} {r['alias'][:10]}", (r["ptei_mean"], r["contra_mean"]),
                    fontsize=8, color="#333", alpha=0.85, va="center")

ax.set_xlabel("PTEI_total 평균 (호재 신호)")
ax.set_ylabel("Contradiction_total 평균 (악재 신호)")
ax.set_title("악재 정책 17건 — PTEI vs Contradict\n대각선 위쪽 = 정책 방향(-)과 일치 / 아래쪽 = 호재로 보도 (모순)")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, max_v)
ax.set_ylim(-0.02, max_v)

plt.tight_layout()
out = "/Users/somv/Documents/Playground/soccz.github.io/projects/market-impact-v2/assets/F34_negative_audit_scatter.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved: {out}")
