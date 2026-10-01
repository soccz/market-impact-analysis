"""Plots use saved result counts, not unrun or inferred performance."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
COLORS = ["#aeb9c8", "#ddaa63", "#147e78"]


def load(name):
    return json.loads((ROOT / name).read_text())


def save(fig, name):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out / (name + "." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    stages = [
        "Previous",
        "Candidate v1",
        "Clarified",
        "Feedback",
        "ALL / ANY",
        "+ fact grammar",
    ]
    for ax, (model, label) in zip(
        axes, [("qwen", "Qwen"), ("kanana-public", "Public Kanana")]
    ):
        vals = []
        d = load("results/explicit_logic/summary.json")
        vals.append(sum(d[s][model]["previous"]["correct"] for s in d))
        for split in ["development", "clarification", "repair", "explicit_logic"]:
            d = load(f"results/{split}/summary.json")
            vals.append(sum(d[s][model]["candidate_model"]["correct"] for s in d))
        d = load("results/aligned/summary.json")
        vals.append(sum(d[s][model]["correct"] for s in d))
        ax.plot(range(6), vals, marker="o", color=COLORS[2])
        ax.set(
            ylim=(-3, 54),
            xticks=range(6),
            xticklabels=stages,
            title=label,
            ylabel="Correct claims / 48",
        )
        ax.tick_params(axis="x", rotation=32)
        for x, y in enumerate(vals):
            ax.annotate(
                str(y), (x, y), xytext=(0, 7), textcoords="offset points", ha="center"
            )
        ax.axhline(48, color=COLORS[0], linestyle="--", linewidth=1)
    fig.suptitle("Observed development: failed feedback was retained", fontsize=14)
    save(fig, "development-path")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for ax, split, title in zip(
        axes,
        ["transfer", "transfer_repair", "confirmation"],
        [
            "First new municipalities (n=20)",
            "Same documents after repair (n=20)",
            "Later new institution (n=10)",
        ],
    ):
        d = load(f"results/{split}/summary.json")
        labels = []
        correct = []
        abstain = []
        wrong = []
        for model, name in [("qwen", "Q"), ("kanana-public", "K")]:
            for method, abbr in [
                ("direct", "direct"),
                ("rules", "rules"),
                ("model", "model"),
            ]:
                x = d[model][method]
                labels.append(name + "/" + abbr)
                correct.append(x["correct"])
                abstain.append(x["abstain"])
                wrong.append(x["n"] - x["correct"] - x["abstain"])
        ax.barh(labels, correct, color=COLORS[2], label="Correct")
        ax.barh(labels, wrong, left=correct, color="#b86050", label="Wrong")
        ax.barh(
            labels,
            abstain,
            left=[a + b for a, b in zip(correct, wrong)],
            color=COLORS[0],
            label="Abstain",
        )
        ax.invert_yaxis()
        ax.set(title=title, xlabel="Claims")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.11), ncol=3)
    save(fig, "transfer-and-coverage")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    d = load("results/revision/summary.json")
    for ax, (model, label) in zip(
        axes, [("qwen", "Qwen"), ("kanana-public", "Public Kanana")]
    ):
        x = d[model]["saved"]
        vals = [
            x["stored_after_correct"],
            x["direct_after_correct"],
            x["raw_pipeline_correct"],
            x["updated_correct"],
        ]
        bars = ax.bar(
            [
                "Keep stored",
                "Direct after",
                "Model slots\n+ contract",
                "Grounded\n+ update",
            ],
            vals,
            color=[COLORS[0], COLORS[0], COLORS[1], COLORS[2]],
        )
        ax.bar_label(bars, padding=3)
        ax.set(ylim=(0, 22), ylabel="Correct after revision / 18", title=label)
    fig.suptitle(
        "Updating changed cases also requires auditing old mistakes", fontsize=14
    )
    save(fig, "revision-chain")


if __name__ == "__main__":
    main()
