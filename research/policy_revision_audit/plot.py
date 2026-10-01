"""Export the observed 18 table queries, without source-document imagery."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / "results/cases.json").read_text())
    groups = ["개인", "단체2~4명", "단체5명이상"]
    labels = ["Individual", "Team 2–4", "Team 5+"]
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    fig.patch.set_facecolor("#fbfaf5")
    ax.set_facecolor("#fbfaf5")
    for ci, competition in enumerate(["전국대회", "국제대회"]):
        for gi, group in enumerate(groups):
            y = ci * 3 + gi
            for rank in [1, 2, 3]:
                row = next(
                    r
                    for r in rows
                    if r["competition"] == competition
                    and r["group"] == group
                    and r["rank"] == rank
                )
                old, new = row["before_thousand_won"], row["after_thousand_won"]
                changed = old != new
                ax.add_patch(
                    plt.Rectangle(
                        (rank - 1.48, y - 0.45),
                        0.96,
                        0.9,
                        facecolor="#d8e9e0" if changed else "#ecebe5",
                        edgecolor="white",
                    )
                )
                ax.text(
                    rank - 1,
                    y - 0.08,
                    f"{old:,} → {new:,}" if changed else f"{old:,} = {new:,}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#193a36",
                )
                ax.text(
                    rank - 1,
                    y + 0.2,
                    "changed" if changed else "unchanged",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#4c625c",
                )
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["1st place", "2nd place", "3rd place"],
        yticks=range(6),
        yticklabels=[
            f"{c} / {g}" for c in ["National", "International"] for g in labels
        ],
        xlim=(-0.5, 2.5),
        ylim=(5.5, -0.5),
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "One erratum, 18 subject-specific table queries",
        fontsize=17,
        weight="bold",
        color="#193a36",
    )
    fig.supxlabel(
        "14 changed · 4 unchanged | Printed amounts in KRW thousands\nOkcheon 2024 scholarship table; exploratory reconstruction, not held-out accuracy.",
        fontsize=10,
    )
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "revision-matrix.png", dpi=180)
    fig.savefig(
        out / "revision-matrix.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
