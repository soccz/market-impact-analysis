"""Export measured summaries, seed ranges, complete case audit and publication figures."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from build_data import RELATIONS, STATES
from evaluate import read, decode, correct

ROOT = Path(__file__).resolve().parent
SEEDS = [17, 42, 2026]
NAMES = {
    "character": "문자 기준선",
    "flat": "24분류",
    "flat_marginal": "24분류 → 축별 선택",
    "factor": "관계·상태 분리",
    "evidence": "분리 + 근거 학습",
}
KO_R = ["오기 정정", "정책 변경", "같은 금액", "관계 불명"]
KO_S = ["확정 서술", "계획", "검토·미확정", "부인", "가정", "상태 불명"]


def values(summary, model, split="evaluation", metric="joint", composition=None):
    keys = (
        ["character:native"]
        if model == "character"
        else [
            f'{"flat" if model=="flat_marginal" else model}-{seed}:{"marginal" if model=="flat_marginal" else "native"}'
            for seed in SEEDS
        ]
    )
    return [
        (
            summary[k]["sets"][split]["compositions"][composition][metric]
            if composition
            else summary[k]["sets"][split]["metrics"][metric]
        )
        for k in keys
    ]


def stats(x):
    return dict(mean=float(np.mean(x)), min=float(min(x)), max=float(max(x)), seeds=x)


def fmt(x):
    return f"{x*100:.2f}%"


def main():
    first = json.loads((ROOT / "results/summary.json").read_text())
    fit = json.loads((ROOT / "results_fit/summary.json").read_text())
    table = {}
    for model in NAMES:
        table[model] = {
            "initial_joint": stats(values(first, model)),
            "fit_joint": stats(values(fit, model)),
            "fit_axes": stats(values(fit, model, metric="axes")),
            "held_axes": stats(
                values(fit, model, metric="axes", composition="held_out")
            ),
            "seen_axes": stats(values(fit, model, metric="axes", composition="seen")),
            "official_axes": stats(values(fit, model, split="official", metric="axes")),
            "official_joint": stats(values(fit, model, split="official")),
            "distractor_joint": stats(values(fit, model, split="distractor")),
            "units_joint": stats(values(fit, model, split="units")),
        }
    records = []
    for model in ["flat", "factor", "evidence"]:
        for seed in SEEDS:
            tag = f"{model}-{seed}:native"
            v = fit[tag]
            records.append(
                dict(
                    model=model,
                    seed=seed,
                    training_joint=v["sets"]["train"]["metrics"]["joint"],
                    evaluation_joint=v["sets"]["evaluation"]["metrics"]["joint"],
                    official_axes=v["sets"]["official"]["metrics"]["axes"],
                    threshold=v["threshold"],
                    accepted=v["sets"]["evaluation"]["selective"]["accepted"],
                    wrong=v["sets"]["evaluation"]["selective"]["wrong"],
                    official_accepted=v["sets"]["official"]["selective"]["accepted"],
                    official_wrong=v["sets"]["official"]["selective"]["wrong"],
                    evidence_character_f1=v["sets"]["evaluation"]["metrics"][
                        "evidence_character_f1"
                    ],
                    distractor_joint=v["sets"]["distractor"]["metrics"]["joint"],
                    unit_joint=v["sets"]["units"]["metrics"]["joint"],
                )
            )
    with (ROOT / "results_fit/comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    rows = read("evaluation")
    matrix = np.zeros((4, 6))
    for seed in SEEDS:
        ps = json.loads(
            (
                ROOT / f"results_fit/evidence-{seed}/evaluation_predictions.json"
            ).read_text()
        )
        for ri, rel in enumerate(RELATIONS):
            for si, state in enumerate(STATES):
                idx = [
                    i
                    for i, r in enumerate(rows)
                    if r["relation"] == rel and r["state"] == state
                ]
                matrix[ri, si] += (
                    np.mean([correct(rows[i], ps[i])["axes"] for i in idx]) / 3
                )
    official = []
    for i, row in enumerate(read("official")):
        predictions = {}
        for model in ["character", "flat", "factor", "evidence"]:
            preds = []
            for seed in [None] if model == "character" else SEEDS:
                name = model if seed is None else f"{model}-{seed}"
                p = decode(
                    json.loads(
                        (
                            ROOT / f"results_fit/{name}/official_predictions.json"
                        ).read_text()
                    )[i]
                )
                preds.append(
                    dict(
                        seed=seed,
                        relation=p["relation"],
                        state=p["state"],
                        confidence=p["confidence"],
                        matches=correct(row, p),
                        spans=p["spans"],
                        evidence=p["evidence"],
                    )
                )
            predictions[model] = preds
        official.append(dict(case=row, predictions=predictions))
    report = dict(
        models=table,
        seed_rows=records,
        evidence_matrix=matrix.tolist(),
        official=official,
        interpretation="Synthetic authored grammar; post-hoc second regime; official AI provisional labels, not independent human gold; seed ranges are not population confidence intervals.",
    )
    (ROOT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    font_manager.fontManager.addfont(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axs[0, 0]
    models = list(NAMES)
    ys = np.arange(len(models))
    a = np.array([table[m]["initial_joint"]["mean"] * 100 for m in models])
    b = np.array([table[m]["fit_joint"]["mean"] * 100 for m in models])
    ax.barh(ys - 0.16, a, height=0.29, color="#b8c3bb", label="첫 학습 조건")
    ax.barh(
        ys + 0.16, b, height=0.29, color="#276348", label="학습 조건 추가 비교 (사후)"
    )
    ax.set_yticks(ys, [NAMES[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("관계·상태·모든 금액 역할 공동 일치 (%)")
    ax.set_title("A. 학습 조건을 먼저 점검하기", loc="left", fontweight="bold", pad=35)
    for y, v in zip(ys, b):
        ax.text(v + 1, y + 0.16, f"{v:.1f}", va="center", fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=8)
    ax = axs[0, 1]
    im = ax.imshow(matrix * 100, vmin=0, vmax=100, cmap="YlGn", aspect="auto")
    ax.set_yticks(range(4), KO_R)
    ax.set_xticks(range(6), ["확정", "계획", "미정", "부인", "가정", "불명"])
    ax.set_title("B. 관계·상태 조합별 동시 일치", loc="left", fontweight="bold")
    for r in range(4):
        for s in range(6):
            held = (r, s) in [(0, 1), (1, 3), (2, 4), (3, 2)]
            ax.text(
                s,
                r,
                f"{matrix[r,s]*100:.0f}" + ("＊" if held else ""),
                ha="center",
                va="center",
                color="white" if matrix[r, s] > 0.65 else "#183628",
            )
    ax.set_xlabel(
        "근거 학습 조건 · 3 seed 평균 · ＊ 학습에서 제외한 조합\n한 칸 24행; 같은 작성 문법의 반복"
    )
    fig.colorbar(im, ax=ax, shrink=0.65, label="%")
    ax = axs[1, 0]
    for j, (key, label, color) in enumerate(
        [
            ("fit_joint", "기본", "#276348"),
            ("distractor_joint", "다른 사업 문장 추가", "#b77734"),
            ("units_joint", "모두 원 단위로 변환", "#637d94"),
        ]
    ):
        vals = [
            table[m][key]["mean"] * 100
            for m in ["character", "flat_marginal", "factor", "evidence"]
        ]
        ax.bar(np.arange(4) + (j - 1) * 0.23, vals, 0.22, label=label, color=color)
    ax.set_xticks(range(4), ["문자", "24분류 축별", "관계·상태", "+ 근거"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("공동 일치 (%)")
    ax.set_title("C. 표면·방해 문장을 바꾼 결과", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax = axs[1, 1]
    for i, m in enumerate(["character", "flat_marginal", "factor", "evidence"]):
        vals = values(fit, m, split="official", metric="axes")
        ax.scatter([i] * len(vals), np.array(vals) * 12, s=40, color="#276348")
        ax.plot([i - 0.18, i + 0.18], [np.mean(vals) * 12] * 2, color="#b77734", lw=3)
    ax.set_xticks(range(4), ["문자", "24분류 축별", "관계·상태", "+ 근거"])
    ax.set_ylim(-0.5, 12.5)
    ax.set_yticks(range(0, 13, 2))
    ax.set_ylabel("관계·상태를 함께 맞힌 수 / 12")
    ax.set_title("D. 공식 구절은 별도의 적용성 점검", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.95,
        "점: 각 seed · 선: 평균\n선정 사례 / AI 임시 라벨 / 사람 검수 전",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.suptitle(
        "정책 문장의 관계·상태·근거 분리 — 실제 실행 결과",
        fontsize=17,
        fontweight="bold",
    )
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for extension in ["svg", "png", "pdf"]:
        path = out / f"factor-status-results.{extension}"
        # Cairo embeds the CFF font correctly and preserves searchable Korean text.
        options = {"backend": "cairo"} if extension == "pdf" else {}
        fig.savefig(path, dpi=170, **options)
        if extension == "svg":
            path.write_text(
                "\n".join(x.rstrip() for x in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    # Full table, including negative first-regime results and all official cases.
    md = [
        "# v3 관측 결과\n",
        "사후 학습 조건 비교 포함. 합성 규칙 정답·공식 AI 임시 라벨이며 사람 검수 전입니다.\n",
        "| 방법 | 첫 조건 공동 일치 | 추가 조건 공동 일치 (seed 범위) | 새 조합 관계·상태 | 공식 12구절 관계·상태 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, d in table.items():
        md.append(
            f"| {NAMES[m]} | {fmt(d['initial_joint']['mean'])} | {fmt(d['fit_joint']['mean'])} ({fmt(d['fit_joint']['min'])}–{fmt(d['fit_joint']['max'])}) | {fmt(d['held_axes']['mean'])} | {d['official_axes']['mean']*12:.2f}/12 |"
        )
    md += [
        "\n공동 일치는 관계·상태·금액 역할 모두를 요구합니다. 공식 열은 관계·상태만이며 서로 다른 과제 조건을 합산하지 않습니다. 추가 조건은 출력층 학습률과 학습량을 함께 바꿨으므로 각각의 효과를 분리한 결과가 아닙니다.\n",
        "## 모든 seed의 응답·오류\n",
        "| 조건 | seed | 학습 공동 | 평가 공동 | 평가 답변/오답 | 공식 답변/오답 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in records:
        md.append(
            f"| {NAMES[x['model']]} | {x['seed']} | {fmt(x['training_joint'])} | {fmt(x['evaluation_joint'])} | {x['accepted']}/{x['wrong']} | {x['official_accepted']}/{x['official_wrong']} |"
        )
    md += [
        "\n응답 문턱은 각 조건의 검증에서만 정했습니다. 0회 답변의 오류율은 0%가 아니라 정의되지 않음(null)입니다. 유형별 응답·오답은 [summary.json](results_fit/summary.json)의 by_class에서 전부 확인할 수 있습니다.\n",
        "## 공식 구절 전체 — seed 42, 근거 학습 조건\n",
        "| 사례 | 임시 관계·상태 | 모델 관계·상태 | 공동 일치 | 출처 |",
        "|---|---|---|---|---|",
    ]
    for x in official:
        row = x["case"]
        p = x["predictions"]["evidence"][1]
        md.append(
            f"| {row['query']} | {row['relation']} / {row['state']} | {p['relation']} / {p['state']} | {'일치' if p['matches']['joint'] else '불일치'} | [{row['publisher']}]({row['url']}) |"
        )
    md += [
        "\n세 초기값의 원시 출력과 모든 공식 사례는 [report.json](report.json)에 있습니다. 특정 seed나 맞힌 사례만 선정해 성능을 표시하지 않습니다. 독립 판독이 완료되면 최초 임시 라벨과 이견을 보존하고 별도의 새 평가로 기록해야 합니다.\n",
        "![실측 비교](figures/factor-status-results.svg)\n",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(md).rstrip() + "\n")
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
