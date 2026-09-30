"""Metrics and fixed validation selection; never use evaluation labels for decoding."""

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from build_data import RELATIONS, STATES

ROOT = Path(__file__).resolve().parent


def read(name):
    return [
        json.loads(s)
        for s in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines()
    ]


def decode(prediction, variant="native"):
    p = deepcopy(prediction)
    rp = np.array(p["relation_probabilities"])
    sp = np.array(p["state_probabilities"])
    if variant in ["marginal", "numeric"]:
        if variant == "numeric":
            before = [s for s in p["spans"] if s["role"] == "before"]
            after = [s for s in p["spans"] if s["role"] == "after"]
            if len(before) == len(after) == 1 and before[0]["won"] != after[0]["won"]:
                rp[2] = 0
                rp = rp / rp.sum()
        p["relation"] = RELATIONS[int(rp.argmax())]
        p["state"] = STATES[int(sp.argmax())]
    confidence = [rp[RELATIONS.index(p["relation"])], sp[STATES.index(p["state"])]]
    confidence += [max(s["probabilities"]) for s in p["spans"]]
    p["confidence"] = float(min(confidence))
    return p


def correct(row, p):
    r = row["relation"] == p["relation"]
    s = row["state"] == p["state"]
    g = [(x["start"], x["end"], x["won"], x["role"]) for x in row["spans"]]
    z = [(x["start"], x["end"], x["won"], x["role"]) for x in p["spans"]]
    role = g == z
    return dict(relation=r, state=s, axes=r and s, roles=role, joint=r and s and role)


def support_set(row, spans):
    return {
        (i, s["kind"])
        for s in spans
        for i in range(s["start"], s["end"])
        if not row["text"][i].isspace()
    }


def metrics(rows, ps):
    assert [x["id"] for x in rows] == [x["id"] for x in ps]
    checks = [correct(r, p) for r, p in zip(rows, ps)]
    out = dict(
        n=len(rows), **{k: float(np.mean([c[k] for c in checks])) for k in checks[0]}
    )
    for key, labels in [("relation", RELATIONS), ("state", STATES)]:
        out[key + "_macro_f1"] = float(
            f1_score(
                [x[key] for x in rows],
                [p[key] for p in ps],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    tp = fp = fn = 0
    for row, p in zip(rows, ps):
        a = support_set(row, row["evidence"])
        b = support_set(row, p["evidence"])
        tp += len(a & b)
        fp += len(b - a)
        fn += len(a - b)
    out["evidence_character_f1"] = (
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    )
    out["evidence_counts"] = dict(tp=tp, fp=fp, fn=fn)
    return out


def selective(rows, ps, tau):
    accepted = [
        i for i, p in enumerate(ps) if tau is not None and p["confidence"] >= tau
    ]
    wrong = sum(not correct(rows[i], ps[i])["joint"] for i in accepted)
    out = dict(
        n=len(rows),
        accepted=len(accepted),
        wrong=wrong,
        coverage=len(accepted) / len(rows),
        risk=wrong / len(accepted) if accepted else None,
    )
    out["by_class"] = {}
    for key, labels in [("relation", RELATIONS), ("state", STATES)]:
        out["by_class"][key] = {}
        for label in labels:
            idx = [i for i, r in enumerate(rows) if r[key] == label]
            ans = [i for i in idx if i in accepted]
            bad = sum(not correct(rows[i], ps[i])["joint"] for i in ans)
            out["by_class"][key][label] = dict(
                n=len(idx),
                accepted=len(ans),
                wrong=bad,
                coverage=len(ans) / len(idx) if idx else None,
                risk=bad / len(ans) if ans else None,
            )
    return out


def choose(rows, ps):
    curve = [
        dict(threshold=float(t), **selective(rows, ps, float(t)))
        for t in np.arange(100) / 100
    ]
    valid = [x for x in curve if x["accepted"] >= 20 and x["risk"] <= 0.05]
    best = max(valid, key=lambda x: (x["accepted"], -x["threshold"])) if valid else None
    return (best["threshold"] if best else None), curve


def main():
    report = {}
    for directory in sorted((ROOT / "results_fit").iterdir()):
        if not directory.is_dir():
            continue
        predictions = {
            p.name.removesuffix("_predictions.json"): json.loads(p.read_text())
            for p in directory.glob("*_predictions.json")
        }
        if "evaluation" not in predictions:
            continue
        variants = ["native", "numeric"] + (
            ["marginal"] if directory.name.startswith("flat-") else []
        )
        for variant in variants:
            tag = directory.name + ":" + variant
            decoded = {
                k: [decode(p, variant) for p in ps] for k, ps in predictions.items()
            }
            tau, curve = choose(read("validation"), decoded["validation"])
            result = dict(threshold=tau, validation_curve=curve, sets={})
            for name, ps in decoded.items():
                rows = read(name)
                if name == "evidence_removed":
                    base = decoded["evaluation"]
                    result["oracle_evidence_removal"] = dict(
                        n=len(ps),
                        axis_decision_retention=float(
                            np.mean(
                                [
                                    (a["relation"], a["state"])
                                    == (b["relation"], b["state"])
                                    for a, b in zip(base, ps)
                                ]
                            )
                        ),
                        confidence_drop=float(
                            np.mean(
                                [
                                    a["confidence"] - b["confidence"]
                                    for a, b in zip(base, ps)
                                ]
                            )
                        ),
                    )
                    continue
                result["sets"][name] = dict(
                    metrics=metrics(rows, ps), selective=selective(rows, ps, tau)
                )
                if name in ["evaluation", "distractor", "units"]:
                    result["sets"][name]["compositions"] = {}
                    for comp in ["seen", "held_out"]:
                        idx = [
                            i for i, r in enumerate(rows) if r["composition"] == comp
                        ]
                        result["sets"][name]["compositions"][comp] = metrics(
                            [rows[i] for i in idx], [ps[i] for i in idx]
                        )
                if name in ["distractor", "units"]:
                    result["sets"][name]["axis_prediction_invariance"] = float(
                        np.mean(
                            [
                                (a["relation"], a["state"])
                                == (b["relation"], b["state"])
                                for a, b in zip(decoded["evaluation"], ps)
                            ]
                        )
                    )
            report[tag] = result
    (ROOT / "results_fit/summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    for key, v in report.items():
        m = v["sets"]["evaluation"]["metrics"]
        o = v["sets"]["official"]["metrics"]
        g = v["sets"]["evaluation"]["selective"]
        print(
            key,
            "axes",
            round(m["axes"], 4),
            "joint",
            round(m["joint"], 4),
            "held",
            round(v["sets"]["evaluation"]["compositions"]["held_out"]["axes"], 4),
            "official",
            round(o["axes"], 4),
            "gate",
            g["accepted"],
            g["wrong"],
        )


if __name__ == "__main__":
    main()
