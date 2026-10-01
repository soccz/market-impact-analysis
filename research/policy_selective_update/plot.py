"""Standalone figures; counts from saved first-response scores, not fitted values."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
NAMES = {
    "refresh_all": "Refresh all",
    "citation_text_gate": "Citation text",
    "hash_audit": "Fixed hash audit",
    "direct_scope_gate": "Direct scope",
    "change_map_gate": "Change map",
    "audited_change_map": "Map + audit",
}


def read(n):
    return json.loads((P / n).read_text())


def save(fig, name):
    p = P / "figures"
    p.mkdir(exist_ok=True)
    fig.savefig(
        p / (name + ".png"), dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    fig.savefig(
        p / (name + ".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main():
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharex=True)
    for i, (split, title) in enumerate(
        [("curated", "Curated clauses"), ("retrieved", "Full-PDF retrieval")]
    ):
        data = read("results/" + split + "/summary.json")
        for j, model in enumerate(["qwen", "kanana-public"]):
            ax = axes[i, j]
            v = data[model]
            methods = list(NAMES)
            ax.barh(
                range(6),
                [v[m]["correct"] for m in methods],
                color="#396b53",
                height=0.6,
            )
            ax.barh(
                range(6),
                [32 - v[m]["correct"] for m in methods],
                left=[v[m]["correct"] for m in methods],
                color="#e5d6ca",
                height=0.6,
            )
            for y, m in enumerate(methods):
                ax.text(
                    0.5,
                    y,
                    f"{v[m]['correct']}/32",
                    color="white" if v[m]["correct"] >= 5 else "#193a36",
                    va="center",
                    fontweight="bold",
                )
            ax.set_yticks(
                range(6), [NAMES[m] + f"  ({v[m]['refreshed']} reads)" for m in methods]
            )
            ax.invert_yaxis()
            ax.set_xlim(0, 32)
            ax.set_xticks([0, 8, 16, 24, 32])
            ax.set_title(title + " / " + ("Qwen" if j == 0 else "Public Kanana"))
            ax.set_xlabel("Correct final judgments; parentheses = fresh reads")
    fig.suptitle(
        "Selective updating can preserve old errors or add routing cost", fontsize=16
    )
    fig.text(
        0.5,
        -0.035,
        "32 shared hypothetical claims from two document families. AI provisional references; no independent human review.\nFresh-read counts exclude gate/map overhead. Context conditions are paired, not independent samples.",
        ha="center",
        fontsize=9,
    )
    save(fig, "selective-updating")
    data = read("results/retrieval_repairs.json")
    keys = ["retrieved", "top5", "neighbor5"]
    labels = ["Top 3", "Top 5", "Top 3 + neighbors"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained", sharex=True)
    for i, model in enumerate([None, "qwen", "kanana-public"]):
        ax = axes[i]
        if model is None:
            vals = [data[k]["qwen"]["metrics"]["supplied_full_span"] for k in keys]
            ax.barh(range(3), vals, color="#859376")
            ax.set_title("Predefined source span supplied")
        else:
            vals = [data[k][model]["metrics"]["correct"] for k in keys]
            joint = [data[k][model]["metrics"]["joint"] for k in keys]
            ax.barh(range(3), vals, color="#d4ddcb", label="Correct")
            ax.barh(range(3), joint, color="#396b53", label="Correct + cited span")
            ax.set_title("Qwen" if model == "qwen" else "Public Kanana")
            ax.legend(fontsize=8, loc="lower right")
        for y, value in enumerate(vals):
            ax.text(value + 0.4, y, str(value), va="center")
        ax.set_yticks(range(3), labels if i == 0 else [])
        ax.invert_yaxis()
        ax.set_xlim(0, 34)
        ax.set_xticks([0, 8, 16, 24, 32])
        ax.set_xlabel("Claims / 32")
    fig.suptitle(
        "Retrieval coverage and reading correctness are different outcomes", fontsize=15
    )
    fig.text(
        0.5,
        -0.06,
        "Same-source exploratory repair. At most five 500-character chunks per version for the two expansions.\nTop 3 is the frozen baseline; missing reference span does not imply that every incorrect result is a reading error.",
        ha="center",
        fontsize=9,
    )
    save(fig, "retrieval-repair")
    data = read("results/contract/summary.json")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for i, m in enumerate(["qwen", "kanana-public"]):
        v = data[m]["summary"]
        ax.bar(
            i - 0.17,
            v["direct_correct"],
            width=0.32,
            color="#b87850",
            label="Same-clause direct read" if i == 0 else None,
        )
        ax.bar(
            i + 0.17,
            v["contract_correct"],
            width=0.32,
            color="#396b53",
            label="Extracted contract + execution" if i == 0 else None,
        )
        for x, y in [
            (i - 0.17, v["direct_correct"]),
            (i + 0.17, v["contract_correct"]),
        ]:
            ax.text(x, y + 0.1, str(y) + "/6", ha="center")
    ax.set_xticks([0, 1], ["Qwen", "Public Kanana"])
    ax.set_ylim(0, 7.5)
    ax.set_ylabel("Correct after-update cap judgments")
    ax.legend(loc="upper center", fontsize=9)
    ax.set_title(
        "A narrow temporal-cap diagnostic, after observing source errors", pad=15
    )
    fig.text(
        0.5,
        -0.06,
        "Manual clause selection and structured query slots. Six observed claims; not an end-to-end language task.\nPublic Kanana old-rule extraction was rejected (0/12); validity does not prove semantic correctness.",
        ha="center",
        fontsize=9,
    )
    save(fig, "temporal-contract")


if __name__ == "__main__":
    main()
