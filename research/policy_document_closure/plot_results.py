"""Standalone figures from saved scores; no inference or reference rewriting."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
DOCS = [
    "auth-exception-any",
    "auth-exception-all",
    "auth-subject-self",
    "auth-subject-spouse",
    "work24",
    "nts",
]
LABELS = [
    "Authored: exception OR",
    "Authored: exception AND",
    "Authored: own income",
    "Authored: spouse income",
    "New source: Work24",
    "New source: NTS",
]
METHODS = ["literal_rules", "guarded_rules", "direct", "full", "retrieved"]
NAMES = [
    "Flat rules",
    "Guarded rules",
    "Direct LLM",
    "Full → graph",
    "Retrieved → graph",
]


def load(name):
    return json.loads((ROOT / name).read_text())


def save(fig, name):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(
            out / (name + "." + ext),
            dpi=180,
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None,
        )
    plt.close(fig)


def main():
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    summary = load("results/summary.json")["all"]
    rows = load("results/rows.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for ax, model, title in zip(
        axes, ["qwen", "kanana-public"], ["Qwen", "Public Kanana"]
    ):
        data = np.array(
            [
                [summary[model][m]["documents"][d]["correct"] for m in METHODS]
                for d in DOCS
            ]
        )
        ax.imshow(data, vmin=0, vmax=8, cmap="YlGnBu", aspect="auto")
        for i, d in enumerate(DOCS):
            rr = [r for r in rows if r["model"] == model and r["document"] == d]
            for j, m in enumerate(METHODS):
                abstain = sum(r[m] == "abstain" for r in rr)
                label = f"{data[i,j]}/8" + (f"\n({abstain} abstain)" if abstain else "")
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if data[i, j] > 5 else "#243346",
                )
        ax.set(
            xticks=range(5),
            xticklabels=NAMES,
            yticks=range(6),
            yticklabels=LABELS,
            title=title,
        )
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle(
        "Document-level outcomes: answer accuracy and abstention remain separate",
        fontsize=13,
    )
    save(fig, "document-matrix")
    retrieval = load("results/retrieval.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    y = np.arange(6)
    axes[0].barh(
        y,
        [retrieval[d]["total_paragraphs"] for d in DOCS],
        color="#dce4e5",
        label="Full body",
    )
    axes[0].barh(
        y,
        [len(retrieval[d]["expanded"]) for d in DOCS],
        color="#247b72",
        label="Retrieved + references",
    )
    axes[0].set(
        yticks=y,
        yticklabels=LABELS,
        xlabel="Paragraphs supplied",
        title="Input reduction, not a correctness guarantee",
    )
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    for offset, model, color in [
        (-0.18, "qwen", "#247b72"),
        (0.18, "kanana-public", "#b86c4d"),
    ]:
        vals = [summary[model][m]["correct"] for m in ["direct", "full", "retrieved"]]
        axes[1].bar(np.arange(3) + offset, vals, width=0.34, color=color, label=model)
        for i, v in enumerate(vals):
            axes[1].text(i + offset, v + 1, str(v), ha="center")
    axes[1].set(
        xticks=range(3),
        xticklabels=["Direct LLM", "Full → graph", "Retrieved → graph"],
        ylim=(0, 55),
        ylabel="Correct / 48 authored cases",
        title="Structure is useful only when extraction works",
    )
    axes[1].legend(frameon=False, loc="upper right")
    save(fig, "retrieval-and-execution")


if __name__ == "__main__":
    main()
