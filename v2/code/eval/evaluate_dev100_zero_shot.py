import argparse
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from transformers import pipeline

from ptei_utils import (
    load_csv,
    load_hypotheses,
    load_policy_keywords,
    relevance_label_from_score,
    relevance_v3_1,
    row_key,
    score_stance,
    write_csv,
)


DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def build_consensus(path_a, path_b):
    rows_a = load_csv(path_a)
    rows_b = load_csv(path_b)
    b_map = {row_key(row): row for row in rows_b}
    consensus = []

    for row_a in rows_a:
        key = row_key(row_a)
        row_b = b_map.get(key)
        if not row_b:
            continue

        a_rel = str(row_a.get("relevance", "")).strip()
        b_rel = str(row_b.get("relevance", "")).strip()
        a_stance = (row_a.get("stance_label", "").strip() or "neutral")
        b_stance = (row_b.get("stance_label", "").strip() or "neutral")
        agreed = a_rel == b_rel and a_stance == b_stance

        out = dict(row_a)
        out["labelerA_relevance"] = a_rel
        out["labelerB_relevance"] = b_rel
        out["labelerA_stance"] = a_stance
        out["labelerB_stance"] = b_stance
        out["consensus_status"] = "agree" if agreed else "disagreement"
        out["consensus_relevance"] = a_rel if agreed else ""
        out["consensus_stance"] = a_stance if agreed else ""
        consensus.append(out)

    return consensus


def metric_block(name, y_true, y_pred):
    if not y_true:
        return {
            "name": name,
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "kappa": None,
        }
    return {
        "name": name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def write_summary(path, metrics, counts, model):
    lines = [
        "# Dev 100 Zero-shot Baseline Evaluation",
        "",
        f"- model: `{model}`",
        f"- consensus agreed rows: `{counts['agree']}`",
        f"- excluded labeler disagreements: `{counts['disagreement']}`",
        f"- evaluated rows in this run: `{counts['evaluated']}`",
        "",
        "| target | n | accuracy | macro-F1 | kappa |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['name']} | {item['n']} | {fmt(item['accuracy'])} | "
            f"{fmt(item['macro_f1'])} | {fmt(item['kappa'])} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tau-rel", type=float, default=0.4)
    parser.add_argument("--labeler-a", default="splits/dev_labels_labelerA.csv")
    parser.add_argument("--labeler-b", default="splits/dev_labels_labelerB.csv")
    parser.add_argument("--consensus-out", default="splits/dev100_consensus_labels.csv")
    parser.add_argument("--pred-out", default="reports/dev100_zero_shot_eval.csv")
    parser.add_argument("--summary-out", default="reports/dev100_zero_shot_eval_summary.md")
    args = parser.parse_args()

    policies = load_policy_keywords()
    hypotheses = load_hypotheses()
    consensus = build_consensus(args.labeler_a, args.labeler_b)
    if args.limit:
        consensus = consensus[: args.limit]

    consensus_fields = list(consensus[0].keys())
    write_csv(args.consensus_out, consensus, consensus_fields)

    print(f"Loading zero-shot model: {args.model}")
    zsc = pipeline("zero-shot-classification", model=args.model, device=args.device)

    results = []
    for idx, row in enumerate(consensus, start=1):
        pn = str(row["pn"])
        text = f"{row.get('title', '')}\n{row.get('content', '')}"
        rel_score, rel_detail = relevance_v3_1(row.get("title", ""), row.get("content", ""), policies[pn])
        pred_relevance = relevance_label_from_score(rel_score)

        if rel_score < args.tau_rel:
            pred_stance = "neutral"
            stance_scores = {"support": 0.0, "contradict": 0.0, "neutral": 1.0}
        else:
            pred_stance, stance_scores = score_stance(zsc, text, hypotheses[pn])

        out = dict(row)
        out.update(
            {
                "rel_score_v3_1": f"{rel_score:.3f}",
                "rel_level_v3_1": rel_detail["level"],
                "rel_hit_v3_1": rel_detail.get("hit", ""),
                "pred_relevance": pred_relevance,
                "pred_relevance_binary": int(pred_relevance > 0),
                "pred_stance": pred_stance,
                "p_support": f"{stance_scores['support']:.6f}",
                "p_contradict": f"{stance_scores['contradict']:.6f}",
                "p_neutral": f"{stance_scores['neutral']:.6f}",
            }
        )
        results.append(out)

        if idx % 10 == 0 or idx == len(consensus):
            print(f"scored {idx}/{len(consensus)}")

    pred_fields = list(results[0].keys())
    write_csv(args.pred_out, results, pred_fields)

    eval_rows = [row for row in results if row["consensus_status"] == "agree"]
    relevant_rows = [row for row in eval_rows if int(row["consensus_relevance"]) > 0]

    y_rel = [int(row["consensus_relevance"]) for row in eval_rows]
    y_rel_pred = [int(row["pred_relevance"]) for row in eval_rows]
    y_rel_bin = [int(value > 0) for value in y_rel]
    y_rel_bin_pred = [int(row["pred_relevance_binary"]) for row in eval_rows]
    y_stance = [row["consensus_stance"] for row in eval_rows]
    y_stance_pred = [row["pred_stance"] for row in eval_rows]
    y_stance_rel = [row["consensus_stance"] for row in relevant_rows]
    y_stance_rel_pred = [row["pred_stance"] for row in relevant_rows]

    metrics = [
        metric_block("relevance 3-class", y_rel, y_rel_pred),
        metric_block("relevance binary rel>0", y_rel_bin, y_rel_bin_pred),
        metric_block("stance all agreed", y_stance, y_stance_pred),
        metric_block("stance consensus rel>0", y_stance_rel, y_stance_rel_pred),
    ]

    counts = Counter(row["consensus_status"] for row in results)
    counts["evaluated"] = len(results)
    write_summary(args.summary_out, metrics, counts, args.model)

    print("\n=== Dev 100 zero-shot baseline ===")
    for item in metrics:
        print(
            f"{item['name']}: n={item['n']} "
            f"acc={fmt(item['accuracy'])} macroF1={fmt(item['macro_f1'])} kappa={fmt(item['kappa'])}"
        )
    print(f"\nSaved: {args.pred_out}")
    print(f"Saved: {args.summary_out}")
    print(f"Saved: {args.consensus_out}")


if __name__ == "__main__":
    main()
