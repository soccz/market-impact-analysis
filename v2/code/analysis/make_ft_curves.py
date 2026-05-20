#!/usr/bin/env python3
"""Fine-tune 학습 곡선 (base vs large) — Part 9.4용."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = ["AppleGothic"]
rcParams["axes.unicode_minus"] = False

base = json.load(open("reports/finetune_eval_v1_base.json"))
large = json.load(open("reports/finetune_eval_v2_large.json"))

def extract(h):
    return [e["epoch"] for e in h], [e["rel_kappa"] for e in h], [e["stance_kappa"] for e in h], [e["train_loss"] for e in h]

b_ep, b_rel, b_st, b_loss = extract(base["history"])
l_ep, l_rel, l_st, l_loss = extract(large["history"])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

# κ curves
axes[0].plot(b_ep, b_rel, "o-", color="#4f7ce6", lw=2, label="base relevance κ")
axes[0].plot(b_ep, b_st, "s-", color="#2d9a6a", lw=2, label="base stance κ")
axes[0].plot(l_ep, l_rel, "o--", color="#c66d3a", lw=2, label="large relevance κ", alpha=0.8)
axes[0].plot(l_ep, l_st, "s--", color="#c44a64", lw=2, label="large stance κ", alpha=0.8)
axes[0].axvline(6, color="#888", ls=":", alpha=0.6)
axes[0].text(6.2, 0.05, "large ep6 best\n→ 이후 overfit", fontsize=9, color="#888")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("Dev 100 Cohen's κ")
axes[0].set_title("Fine-tune 학습 곡선 — base 5ep vs large 10ep")
axes[0].legend(loc="lower right", fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(-0.05, 0.7)

# train loss
axes[1].plot(b_ep, b_loss, "o-", color="#4f7ce6", lw=2, label="base train loss")
axes[1].plot(l_ep, l_loss, "o--", color="#c66d3a", lw=2, label="large train loss")
axes[1].axhline(0.1, color="#c44a64", ls=":", alpha=0.5)
axes[1].text(7, 0.13, "0.1 이하 = 학습 데이터 암기 영역", fontsize=9, color="#c44a64")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("Train loss")
axes[1].set_title("학습 손실 — large는 ep8부터 데이터 암기 진입")
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
out = "/Users/somv/Documents/Playground/soccz.github.io/projects/market-impact-v2/assets/F33_finetune_curves.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
plt.close()
print(f"saved: {out}")
