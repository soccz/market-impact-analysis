"""Standalone figures from saved, provisional reference scores."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).parent
OUT = P / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "#faf9f6",
    }
)
COLORS = ["#596777", "#167c80", "#b47439", "#874e70"]


def read(s):
    return json.loads((P / f"results/{s}/summary.json").read_text())


def save(fig, name, caption):
    fig.text(0.04, 0.018, caption, fontsize=8, color="#51545b")
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 4.7))
d = read("official")
ys = np.arange(2)
methods = ["direct", "pipeline", "reference_policy", "reference_profile"]
labels = [
    "Direct reading",
    "Extract + execute",
    "Reference policy (diagnostic)",
    "Reference profile (diagnostic)",
]
for i, (k, l) in enumerate(zip(methods, labels)):
    v = [d[m][k]["correct"] for m in ["qwen", "kanana-public"]]
    yy = ys + (i - 1.5) * 0.18
    ax.barh(yy, v, height=0.16, color=COLORS[i], label=l)
    for y, n in zip(yy, v):
        ax.text(n + 0.35, y, f"{n}/32", va="center", fontsize=9)
ax.set(
    yticks=ys,
    yticklabels=["Qwen", "Public Kanana"],
    xlim=(0, 36),
    xlabel="Claims matching provisional reference / 32",
    title="New-document test: separate rule errors from applicant-fact errors",
)
ax.legend(loc="lower right", fontsize=8)
save(
    fig,
    "primary-diagnosis",
    "Two documents; selected scopes; AI-assisted reference; 0 independent human reviewers.\nReference replacements diagnose errors; they are not deployable model performance.",
)
fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True)
for ax, sp, title, n in zip(
    axes,
    ["development_explicit", "official", "composition"],
    ["Development", "New documents", "New authored combinations"],
    [24, 32, 32],
):
    example = (
        "development_examples" if sp.startswith("development") else sp + "_examples"
    )
    a, b = read(sp), read(example)
    for j, (m, label) in enumerate([("qwen", "Qwen"), ("kanana-public", "Kanana")]):
        vals = [a[m]["pipeline"]["correct"], b[m]["pipeline"]["correct"]]
        ax.plot(
            [0, 1],
            [v / n * 100 for v in vals],
            marker="o",
            color=COLORS[j],
            label=label,
        )
        for x, v in enumerate(vals):
            ax.annotate(
                f"{v}/{n}",
                (x, v / n * 100),
                xytext=(
                    0,
                    (
                        9
                        if vals[x]
                        >= (
                            [a, b][x]["kanana-public" if j == 0 else "qwen"][
                                "pipeline"
                            ]["correct"]
                        )
                        else -18
                    ),
                ),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Explicit task", "+ Logic examples"],
        xlim=(-0.4, 1.4),
        ylim=(0, 100),
        title=title,
    )
axes[0].set_ylabel("Reference match (%)")
axes[2].legend(loc="upper right")
save(
    fig,
    "method-transfer",
    "Same saved direct/profile responses reused within each comparison. Examples were added after document collection.\nDifferent denominators are shown; authored combinations are not an independent policy population.",
)
d = json.loads((P / "results/diagnostics.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, b, title in zip(
    axes, ["cap-rent", "cap-education"], ["Rent conversion", "Education exceptions"]
):
    for j, (m, label) in enumerate([("qwen", "Qwen"), ("kanana-public", "Kanana")]):
        vs = [
            d[s][m]["by_bundle"][b]["pipeline"]
            for s in ["capability_base", "capability_extended", "capability_repair"]
        ]
        ax.plot(range(3), vs, marker="o", label=label, color=COLORS[j])
        for x, y in enumerate(vs):
            ax.annotate(
                f"{y}/8",
                (x, y),
                xytext=(0, 8 if j == 0 else -16),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    ax.set(
        xticks=range(3),
        xticklabels=["Base", "Extended", "+ Feedback"],
        ylim=(-0.9, 9),
        xlim=(-0.3, 2.3),
        title=title,
    )
axes[0].set_ylabel("Claims matching reference / 8")
axes[1].legend(loc="upper right")
save(
    fig,
    "capability-repair",
    "Post-hoc study on the same two documents; extension also changes prompt, fields and output budget.\nOne retry selected by validation, without reference answers. Feedback can reduce correctness.",
)
print("3 PNG + 3 PDF figures generated.")
