"""Fixed metrics and paired synthetic diagnostics; every seed is retained."""

import json
import numpy as np
from common import ROOT, V3, datasets, check_lock, save
from evaluate import decode, correct, metrics, choose, selective

CONDITIONS = ["raw", "raw_scope", "normalized", "normalized_scope"]
SEEDS = [17, 42, 2026]


def average(values):
    return dict(
        mean=float(np.mean(values)),
        min=float(min(values)),
        max=float(max(values)),
        values=values,
    )


def signature(p):
    return p["relation"], p["state"], [(s["role"], s["won"]) for s in p["spans"]]


def main():
    check_lock()
    results, paired, row_scores = {}, {}, {}
    for condition in CONDITIONS:
        rows = datasets(condition.endswith("scope"))
        paired[condition] = {}
        for seed in SEEDS:
            folder = ROOT / "results" / condition / str(seed)
            raw = {n: json.loads((folder / f"{n}.json").read_text()) for n in rows}
            paired[condition][str(seed)] = {}
            for a, b in [("evaluation", "units"), ("distractor", "combined")]:
                paired[condition][str(seed)][a + ":" + b] = sum(
                    signature(x) == signature(y) for x, y in zip(raw[a], raw[b])
                ) / len(rows[a])
            for variant in ["native", "numeric"]:
                decoded = {n: [decode(p, variant) for p in ps] for n, ps in raw.items()}
                tau, curve = choose(rows["validation"], decoded["validation"])
                sets = {
                    n: dict(
                        metrics=metrics(rows[n], ps),
                        selective=selective(rows[n], ps, tau),
                    )
                    for n, ps in decoded.items()
                }
                eval_rows = rows["evaluation"]
                subsets = {}
                for name in ["seen", "held_out"]:
                    ids = [
                        i for i, r in enumerate(eval_rows) if r["composition"] == name
                    ]
                    subsets[name] = metrics(
                        [eval_rows[i] for i in ids],
                        [decoded["evaluation"][i] for i in ids],
                    )
                key = f"{condition}/{seed}:{variant}"
                results[key] = dict(
                    threshold=tau,
                    validation_curve=curve,
                    sets=sets,
                    compositions=subsets,
                )
                row_scores[key] = [
                    int(correct(g, p)["joint"])
                    for g, p in zip(rows["combined"], decoded["combined"])
                ]
    table = {}
    for condition in CONDITIONS:
        table[condition] = {}
        for variant in ["native", "numeric"]:
            table[condition][variant] = {}
            for split in datasets():
                table[condition][variant][split] = {
                    metric: average(
                        [
                            results[f"{condition}/{seed}:{variant}"]["sets"][split][
                                "metrics"
                            ][metric]
                            for seed in SEEDS
                        ]
                    )
                    for metric in [
                        "joint",
                        "axes",
                        "roles",
                        "relation_macro_f1",
                        "state_macro_f1",
                        "evidence_character_f1",
                    ]
                }
    # Prespecified pair comparison, clustered by six value/query frames rather than 576 rows.
    data = datasets()["combined"]
    delta = np.mean(
        [
            np.array(row_scores[f"normalized_scope/{s}:native"])
            - np.array(row_scores[f"raw_scope/{s}:native"])
            for s in SEEDS
        ],
        axis=0,
    )
    frame_scores = [
        float(np.mean([d for r, d in zip(data, delta) if r["frame"] == frame]))
        for frame in range(6)
    ]
    rng = np.random.default_rng(1729)
    samples = np.mean(rng.choice(frame_scores, size=(10000, 6), replace=True), axis=1)
    bootstrap = dict(
        comparison="normalized_scope - raw_scope, combined, native joint",
        mean=float(delta.mean()),
        frame_differences=frame_scores,
        interval95=np.quantile(samples, [0.025, 0.975]).tolist(),
        clusters=6,
        repeats=10000,
        limitation="Conditional synthetic frame variability after mean over seeds; not independent language/event sampling or population uncertainty.",
    )
    save(
        ROOT / "summary.json",
        dict(
            runs=results,
            table=table,
            consistency=paired,
            paired_frame_bootstrap=bootstrap,
        ),
    )
    for condition in CONDITIONS:
        print(
            condition,
            {
                n: round(table[condition]["native"][n]["joint"]["mean"] * 100, 2)
                for n in ["evaluation", "units", "distractor", "combined"]
            },
            "official axes",
            round(table[condition]["native"]["official"]["axes"]["mean"] * 12, 2),
        )


if __name__ == "__main__":
    main()
