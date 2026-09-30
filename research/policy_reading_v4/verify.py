"""Audit every reported metric, restored span, paired prediction and reused model."""

from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
from common import ROOT, V3, datasets, check_lock, save
from representation import money_view

spec = importlib.util.spec_from_file_location("v3_independent_audit", V3 / "verify.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    lock = check_lock()
    summary = json.loads((ROOT / "summary.json").read_text())
    checked, reuse, steps = 0, [], {}
    conditions = ["raw", "raw_scope", "normalized", "normalized_scope"]
    for condition in conditions:
        data = datasets(condition.endswith("scope"))
        for seed in [17, 42, 2026]:
            folder = ROOT / "results" / condition / str(seed)
            predictions = {
                n: json.loads((folder / f"{n}.json").read_text()) for n in data
            }
            training = json.loads((folder / "training.json").read_text())
            if condition.startswith("raw"):
                source = (
                    V3
                    / (
                        "results_augmentation/scope"
                        if condition.endswith("scope")
                        else "results_fit"
                    )
                    / f"evidence-{seed}"
                )
                for name in data:
                    old = source / f"{name}_predictions.json"
                    if old.exists():
                        assert predictions[name] == json.loads(old.read_text()), (
                            condition,
                            seed,
                            name,
                            "reused predictions differ",
                        )
                        reuse.append(f"{condition}/{seed}/{name}")
            else:
                assert not training["reused"] and len(training["steps"]) == 300
                assert all(math.isfinite(x["loss"]) for x in training["steps"])
                steps[f"{condition}/{seed}"] = dict(
                    attempted=300, skipped=sum(x["skipped"] for x in training["steps"])
                )
            for name, rows in data.items():
                raw = predictions[name]
                assert [r["id"] for r in rows] == [p["id"] for p in raw]
                for r, p in zip(rows, raw):
                    audit.probability(p["relation_probabilities"], 4)
                    audit.probability(p["state_probabilities"], 6)
                    for span in p["spans"] + p["evidence"]:
                        assert 0 <= span["start"] < span["end"] <= len(r["text"])
                        assert r["text"][span["start"] : span["end"]] == span["text"]
                    detected = money_view(r["text"])["amounts"]
                    assert [
                        {k: s[k] for k in ["start", "end", "text", "won"]}
                        for s in p["spans"]
                    ] == detected
                    for span in p["spans"]:
                        audit.probability(span["probabilities"], 3)
                for variant in ["native", "numeric"]:
                    result = summary["runs"][f"{condition}/{seed}:{variant}"]
                    ds, es = audit.audit_metrics(
                        rows, raw, variant, result["sets"][name]["metrics"]
                    )
                    audit.audit_gate(
                        rows,
                        ds,
                        es,
                        result["threshold"],
                        result["sets"][name]["selective"],
                    )
                    checked += 1
            for a, b in [("evaluation", "units"), ("distractor", "combined")]:
                consistent = 0
                for x, y in zip(predictions[a], predictions[b]):
                    signature = lambda p: (
                        p["relation"],
                        p["state"],
                        [(s["role"], s["won"]) for s in p["spans"]],
                    )
                    consistent += signature(x) == signature(y)
                    if condition.startswith("normalized"):
                        assert (
                            x["relation_probabilities"] == y["relation_probabilities"]
                        )
                        assert x["state_probabilities"] == y["state_probabilities"]
                        assert [s["probabilities"] for s in x["spans"]] == [
                            s["probabilities"] for s in y["spans"]
                        ]
                audit.close(
                    consistent / len(predictions[a]),
                    summary["consistency"][condition][str(seed)][a + ":" + b],
                )
            for variant in ["native", "numeric"]:
                result = summary["runs"][f"{condition}/{seed}:{variant}"]
                # Select threshold again from independently computed validation decisions.
                raw = predictions["validation"]
                rows = data["validation"]
                ds = [audit.decision(p, variant) for p in raw]
                valid = []
                for t in range(100):
                    idx = [i for i, d in enumerate(ds) if d[2] >= t / 100]
                    wrong = sum(
                        not audit.exact(rows[i], raw[i], ds[i][0], ds[i][1])["joint"]
                        for i in idx
                    )
                    if len(idx) >= 20 and wrong / len(idx) <= 0.05:
                        valid.append((len(idx), -t / 100))
                tau = -max(valid)[1] if valid else None
                assert tau == result["threshold"]
                for name in ["seen", "held_out"]:
                    idx = [
                        i
                        for i, row in enumerate(data["evaluation"])
                        if row["composition"] == name
                    ]
                    audit.audit_metrics(
                        [data["evaluation"][i] for i in idx],
                        [predictions["evaluation"][i] for i in idx],
                        variant,
                        result["compositions"][name],
                    )
    for condition, variants in summary["table"].items():
        for variant, splits in variants.items():
            for split, metrics in splits.items():
                for metric, stats in metrics.items():
                    vals = [
                        summary["runs"][f"{condition}/{s}:{variant}"]["sets"][split][
                            "metrics"
                        ][metric]
                        for s in [17, 42, 2026]
                    ]
                    assert stats["values"] == vals
                    audit.close(sum(vals) / 3, stats["mean"])
                    assert stats["min"] == min(vals) and stats["max"] == max(vals)
    assert len(reuse) == 36
    combined = datasets()["combined"]
    delta = np.zeros(len(combined))
    errors = json.loads((ROOT / "error_groups.json").read_text())
    for condition, sign in [("raw_scope", -1), ("normalized_scope", 1)]:
        counts = Counter()
        for seed in [17, 42, 2026]:
            root = ROOT / "results" / condition / str(seed)
            for i, (g, p) in enumerate(
                zip(combined, json.loads((root / "combined.json").read_text()))
            ):
                r, s, _ = audit.decision(p, "native")
                delta[i] += sign * int(audit.exact(g, p, r, s)["joint"]) / 3
            for g, p in zip(
                datasets()["distractor"],
                json.loads((root / "distractor.json").read_text()),
            ):
                counts.update(
                    (a["role"], b["role"]) for a, b in zip(g["spans"], p["spans"])
                )
        labels = errors["roles"]
        assert [[counts[a, b] for b in labels] for a in labels] == errors["matrices"][
            condition
        ]
    clustered = [
        float(np.mean([delta[i] for i, r in enumerate(combined) if r["frame"] == f]))
        for f in range(6)
    ]
    bootstrap = summary["paired_frame_bootstrap"]
    audit.close(float(np.mean(delta)), bootstrap["mean"])
    for a, b in zip(clustered, bootstrap["frame_differences"]):
        audit.close(a, b)
    samples = (
        np.random.default_rng(1729)
        .choice(clustered, (10000, 6), replace=True)
        .mean(axis=1)
    )
    for a, b in zip(np.quantile(samples, [0.025, 0.975]), bootstrap["interval95"]):
        audit.close(a, b)
    outcome = dict(
        status="PASS",
        locked_files=len(lock["files"]),
        metric_sets=checked,
        reused_prediction_files_equal=len(reuse),
        new_fits=steps,
        normalized_pair_probabilities="Exact equality for all 576+576 paired passages per seed/condition",
        human_reviews_completed=0,
        composition_metrics_and_frame_bootstrap="PASS",
        error_confusions_recomputed="PASS",
        limit="Mechanical validity; no independent human gold or real-world generalization proof.",
    )
    save(ROOT / "verification.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
