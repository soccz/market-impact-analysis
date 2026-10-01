"""Export observed correctness, failures and inference cost without source text."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COLORS = ["#2d7051", "#ba6944", "#b8beb7"]


def load(name):
    return json.loads((ROOT / name).read_text())


def save(fig, name):
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(
        folder / (name + ".png"),
        dpi=180,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    fig.savefig(
        folder / (name + ".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
        bbox_inches="tight",
    )
    plt.close(fig)


def stress(title, rows, n, name, footnote):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4.8 + max(0, len(rows) - 4) * 0.6),
        layout="constrained",
        sharey=True,
    )
    fig.patch.set_facecolor("#fafaf5")
    for i, (variant, label) in enumerate(
        [
            (0, "Relevant clauses"),
            (1, "Other-program distractors"),
            (2, "Evidence withheld"),
        ]
    ):
        ax = axes[i]
        ax.set_facecolor("#fafaf5")
        for y, (method, values) in enumerate(rows):
            m = values[variant]
            parts = [m["correct"], m["wrong_answer"], m["abstain"]]
            left = 0
            assert sum(parts) == n
            for v, color in zip(parts, COLORS):
                ax.barh(y, v, left=left, color=color, height=0.55)
                if v:
                    ax.text(
                        left + v / 2,
                        y,
                        str(v),
                        ha="center",
                        va="center",
                        color="white" if color != COLORS[2] else "#26392e",
                        fontsize=10,
                    )
                left += v
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, n)
        ax.set_xticks([0, n // 2, n])
        ax.set_xlabel(f"Claims (n={n})")
        ax.set_title(label, fontsize=11, pad=18)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(title, fontsize=16, y=1.06)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(
        handles,
        [
            "Correct, valid output",
            "Wrong, valid output",
            "Invalid output / abstention",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=3,
        frameon=False,
    )
    fig.text(0.5, -0.15, footnote, ha="center", va="top", fontsize=9, color="#53645a")
    save(fig, name)


def main():
    first = load("results/summary.json")
    ext = load("results/extension/summary.json")
    citation = load("results/citation/summary.json")
    stress(
        "Jeju: a cleaner explanation is not robust reading",
        [
            (
                label,
                [
                    source["official_" + v][method]
                    for v in ["base", "distractor", "removed"]
                ],
            )
            for source, method, label in [
                (first, "qwen-baseline", "Qwen baseline"),
                (first, "qwen-scoped", "Qwen structured"),
                (ext, "qwen-evidence", "Qwen evidence-first"),
                (citation, "qwen-citation-only", "Qwen citation-only"),
            ]
        ],
        18,
        "jeju-evidence-stress",
        "18 related claims from one revision pair; evidence-first and citation-only are post-hoc. AI-assisted references.",
    )
    stress(
        "Incheon: transfer and a post-hoc citation ablation",
        [
            (
                label,
                [
                    citation["replication_" + v][method]
                    for v in ["base", "distractor", "removed"]
                ],
            )
            for method, label in [
                ("qwen-baseline", "Qwen baseline"),
                ("qwen-evidence", "Qwen evidence-first"),
                ("qwen-citation-only", "Qwen citation-only (post-hoc)"),
                ("kanana-public-baseline", "Public Kanana baseline"),
                ("kanana-public-evidence", "Public Kanana evidence-first"),
                (
                    "kanana-public-citation-only",
                    "Public Kanana citation-only (post-hoc)",
                ),
            ]
        ],
        12,
        "incheon-transfer",
        "12 related claims from one new institution; not 36 independent documents. AI-assisted provisional references.",
    )
    thinking = load("results/thinking/summary.json")["followup"]
    sampling = load("results/sampling/summary.json")["followup"]
    tr = load("results/thinking/runtime.json")
    sr = load("results/sampling/runtime.json")
    rows = [
        (
            "2048 / no thinking",
            thinking["qwen-budget-control"],
            tr["followup/qwen-budget-control"],
        ),
        (
            "2048 / thinking requested",
            thinking["qwen-thinking"],
            tr["followup/qwen-thinking"],
        ),
        (
            "4096 / sampled control",
            sampling["qwen-sampled-control"],
            sr["qwen-sampled-control"],
        ),
        (
            "4096 / sampled thinking",
            sampling["qwen-sampled-thinking"],
            sr["qwen-sampled-thinking"],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    fig.patch.set_facecolor("#fafaf5")
    for ax in axes:
        ax.set_facecolor("#fafaf5")
        ax.set_yticks(range(4), [r[0] for r in rows])
        ax.invert_yaxis()
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    for y, (_, m, r) in enumerate(rows):
        axes[0].barh(y, m["correct"], color=COLORS[0])
        axes[0].text(m["correct"] + 0.15, y, f'{m["correct"]}/12', va="center")
        axes[1].barh(y, r["median_elapsed_seconds"], color="#4c7185")
        axes[1].text(
            r["median_elapsed_seconds"] + 0.5,
            y,
            f'{r["median_elapsed_seconds"]:.2f}s | truncated {r["truncated"]}',
            va="center",
            fontsize=9,
        )
    axes[0].set_xlim(0, 14)
    axes[0].set_title("Correct answers with valid output")
    axes[0].set_xlabel("Twelve follow-up claims")
    axes[1].set_xlim(0, max(r[2]["median_elapsed_seconds"] for r in rows) * 1.4 + 5)
    axes[1].set_title("Observed wall time and budget failures")
    axes[1].set_xlabel("Median observed request time (seconds)")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle("Inference cost does not guarantee a usable answer", fontsize=16)
    fig.text(
        0.5,
        -0.035,
        "Same previously-seen documents. Matched settings within each pair; adaptive sampling/budget change between pairs. One draw per case.",
        ha="center",
        fontsize=9,
    )
    save(fig, "inference-budget")


if __name__ == "__main__":
    main()
