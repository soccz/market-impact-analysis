"""Publication figures from saved counts, without treating variants as independent."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
COLORS = ["#b9c3d2", "#127b79"]


def read(path):
    return json.loads((ROOT / path).read_text())


def finish(fig, name):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#ffffff",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i, (sp, title) in enumerate(
        [("metamorphic", "Authored controls"), ("official", "New scoped documents")]
    ):
        for j, (model, label) in enumerate(
            [("qwen", "Qwen"), ("kanana-public", "Public Kanana")]
        ):
            d = read(f"results/{sp}/summary.json")[model]
            n = d["raw"]["n"]
            ax = axes[i, j]
            before = [d["raw"]["correct"], d["raw_profile_exact"], d["joint_raw"]]
            after = [
                d["corrected"]["correct"],
                d["corrected_profile_exact"],
                d["joint_corrected"],
            ]
            for offset, counts, color, l in [
                (-0.18, before, COLORS[0], "Raw"),
                (0.18, after, COLORS[1], "Grounded"),
            ]:
                bars = ax.bar(
                    [x + offset for x in range(3)], counts, 0.34, color=color, label=l
                )
                ax.bar_label(bars, padding=3)
            ax.set(
                xticks=range(3),
                xticklabels=["Answer", "All facts + polarity", "Both"],
                ylim=(0, n * 1.18),
                title=f"{title} / {label} (n={n})",
                ylabel="Correct claims",
            )
            if i == 0 and j == 0:
                ax.legend(
                    frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2
                )
    fig.suptitle("An unchanged answer can hide a repaired fact", fontsize=15, y=1.03)
    fig.tight_layout()
    finish(fig, "answer-and-facts")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (sp, title) in zip(
        axes,
        [
            ("metamorphic", "Authored controls (n=32)"),
            ("official", "New documents (n=16)"),
        ],
    ):
        labels = []
        ok = []
        wrong = []
        abst = []
        for m, l in [("qwen", "Qwen"), ("kanana-public", "Kanana")]:
            d = read(f"results/{sp}/summary.json")[m]
            for method in ["raw", "reject", "corrected"]:
                x = d[method]
                labels.append(l + " / " + method)
                ok.append(x["correct"])
                abst.append(x["abstain"])
                wrong.append(x["n"] - x["correct"] - x["abstain"])
        ax.barh(labels, ok, color=COLORS[1], label="Correct")
        ax.barh(labels, wrong, left=ok, color="#bd6556", label="Wrong")
        ax.barh(
            labels,
            abst,
            left=[a + b for a, b in zip(ok, wrong)],
            color="#dadde2",
            label="Abstain",
        )
        ax.set(title=title, xlabel="Claims")
        ax.invert_yaxis()
    axes[0].legend(frameon=False, bbox_to_anchor=(0, 1.26), loc="upper left", ncol=3)
    fig.tight_layout()
    finish(fig, "rejection-tradeoff")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pairs = read("results/metamorphic/pairs.json")
    labels = []
    raw = []
    fixed = []
    for m, l in [("qwen", "Qwen"), ("kanana-public", "Kanana")]:
        for v in ["preserve", "change", "unknown"]:
            rows = [r for r in pairs if r["model"] == m and r["variant"] == v]
            labels.append(l + " / " + v)
            raw.append(sum(r["methods"]["raw"]["both_correct"] for r in rows))
            fixed.append(sum(r["methods"]["corrected"]["both_correct"] for r in rows))
    for off, vals, color, label in [
        (-0.18, raw, COLORS[0], "Raw"),
        (0.18, fixed, COLORS[1], "Grounded"),
    ]:
        bars = axes[0].barh(
            [x + off for x in range(6)], vals, 0.34, color=color, label=label
        )
        axes[0].bar_label(bars, padding=2)
    axes[0].set(
        yticks=range(6),
        yticklabels=labels,
        xlim=(0, 9),
        title="Both answers correct, out of 8 pairs",
        xlabel="Pairs",
    )
    axes[0].invert_yaxis()
    axes[0].legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2
    )
    w = read("results/counterexamples_audited.json")
    counts = [
        sum(x["status"] == s for x in w)
        for s in ["witness", "no_witness_in_domain", "unavailable"]
    ]
    bars = axes[1].barh(
        ["Disagreement found", "None in bounded domain", "Invalid / unavailable"],
        counts,
        color=["#bd6556", COLORS[1], "#dadde2"],
    )
    axes[1].bar_label(bars, padding=3)
    axes[1].set(
        xlim=(0, 24), title="Prior extracted rules (n=36)", xlabel="Rule outputs"
    )
    axes[1].invert_yaxis()
    fig.tight_layout()
    finish(fig, "pairs-and-counterexamples")


if __name__ == "__main__":
    main()
