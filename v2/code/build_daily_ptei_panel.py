import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


INPUT = "data/processed/article_scores_hybrid.csv"
OUTPUT = "data/processed/daily_ptei_panel.csv"
CHANNELS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main():
    groups = {}
    with open(INPUT, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["policy_n"],
                row["policy_alias"],
                row["policy_industry"],
                row["policy_d0"],
                row["rel_day"],
                row["pub_date"],
            )
            if key not in groups:
                groups[key] = {
                    "article_count": 0,
                    "provider_counts": Counter(),
                    "ptei_sum": 0.0,
                    "contradiction_sum": 0.0,
                    "active_count": 0,
                    "stance_counts": Counter(),
                    "channel_ptei_sum": defaultdict(float),
                    "channel_contradiction_sum": defaultdict(float),
                    "channel_gate_count": Counter(),
                }
            g = groups[key]
            g["article_count"] += 1
            g["provider_counts"][row.get("provider", "")] += 1
            g["ptei_sum"] += fnum(row.get("PTEI_total"))
            g["contradiction_sum"] += fnum(row.get("Contradiction_total"))
            g["stance_counts"][row.get("pred_stance", "")] += 1
            if row.get("active_channels"):
                g["active_count"] += 1
            for channel in CHANNELS:
                g["channel_ptei_sum"][channel] += fnum(row.get(f"{channel}_ptei"))
                g["channel_contradiction_sum"][channel] += fnum(row.get(f"{channel}_contradiction"))
                g["channel_gate_count"][channel] += int(row.get(f"{channel}_gate") or 0)

    out_rows = []
    for key, g in groups.items():
        policy_n, alias, industry, policy_d0, rel_day, pub_date = key
        n = g["article_count"]
        max_provider_share = max(g["provider_counts"].values()) / n if n else 0.0
        diffusion = 1.0 - max_provider_share if n else 0.0
        log_n = math.log1p(n)
        mean_ptei = g["ptei_sum"] / n if n else 0.0
        mean_contradiction = g["contradiction_sum"] / n if n else 0.0
        row = {
            "policy_n": policy_n,
            "policy_alias": alias,
            "policy_industry": industry,
            "policy_d0": policy_d0,
            "rel_day": rel_day,
            "pub_date": pub_date,
            "article_count": n,
            "provider_count": len(g["provider_counts"]),
            "max_provider_share": f"{max_provider_share:.6f}",
            "diffusion": f"{diffusion:.6f}",
            "active_article_count": g["active_count"],
            "pred_support_count": g["stance_counts"]["support"],
            "pred_contradict_count": g["stance_counts"]["contradict"],
            "pred_neutral_count": g["stance_counts"]["neutral"],
            "PTEI_sum": f"{g['ptei_sum']:.6f}",
            "PTEI_mean": f"{mean_ptei:.6f}",
            "Contradiction_sum": f"{g['contradiction_sum']:.6f}",
            "Contradiction_mean": f"{mean_contradiction:.6f}",
            "NP_support_total": f"{mean_ptei * log_n * diffusion:.6f}",
            "NP_contradict_total": f"{mean_contradiction * log_n * diffusion:.6f}",
        }
        for channel in CHANNELS:
            channel_mean = g["channel_ptei_sum"][channel] / n if n else 0.0
            channel_contra_mean = g["channel_contradiction_sum"][channel] / n if n else 0.0
            row[f"{channel}_gate_count"] = g["channel_gate_count"][channel]
            row[f"{channel}_PTEI_sum"] = f"{g['channel_ptei_sum'][channel]:.6f}"
            row[f"{channel}_PTEI_mean"] = f"{channel_mean:.6f}"
            row[f"{channel}_NP_support"] = f"{channel_mean * log_n * diffusion:.6f}"
            row[f"{channel}_Contradiction_sum"] = f"{g['channel_contradiction_sum'][channel]:.6f}"
            row[f"{channel}_Contradiction_mean"] = f"{channel_contra_mean:.6f}"
            row[f"{channel}_NP_contradict"] = f"{channel_contra_mean * log_n * diffusion:.6f}"
        out_rows.append(row)

    out_rows.sort(key=lambda r: (int(r["policy_n"]), int(r["rel_day"]), r["pub_date"]))
    fieldnames = list(out_rows[0].keys())
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"saved: {OUTPUT}")
    print(f"rows: {len(out_rows)}")
    print(f"policies: {len(set(row['policy_n'] for row in out_rows))}")
    print(f"ptei_sum: {sum(fnum(row['PTEI_sum']) for row in out_rows):.3f}")
    print(f"contradiction_sum: {sum(fnum(row['Contradiction_sum']) for row in out_rows):.3f}")


if __name__ == "__main__":
    main()
