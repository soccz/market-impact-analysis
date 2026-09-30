"""Post-hoc 2x2 training-data intervention; all seeds and unchanged tests."""

from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
SEEDS = [17, 42, 2026]
NAMES = {
    "control": "보강 없음",
    "units": "단위 표기만",
    "scope": "대상 문맥만",
    "both": "둘 다 보강",
}


def main():
    summaries = {"control": json.loads((ROOT / "results_fit/summary.json").read_text())}
    for name in ["units", "scope", "both"]:
        summaries[name] = json.loads(
            (ROOT / f"results_augmentation/{name}/summary.json").read_text()
        )
    table = {}
    diagnostics = {}
    for name, summary in summaries.items():
        table[name] = {}
        for split in ["evaluation", "units", "distractor", "official"]:
            table[name][split] = {}
            for metric in ["axes", "roles", "joint"]:
                values = [
                    summary[f"evidence-{seed}:native"]["sets"][split]["metrics"][metric]
                    for seed in SEEDS
                ]
                table[name][split][metric] = dict(
                    mean=float(np.mean(values)),
                    min=min(values),
                    max=max(values),
                    seeds=values,
                )
        table[name]["held_axes"] = [
            summary[f"evidence-{seed}:native"]["sets"]["evaluation"]["compositions"][
                "held_out"
            ]["axes"]
            for seed in SEEDS
        ]
        table[name]["numeric_joint"] = [
            summary[f"evidence-{seed}:numeric"]["sets"]["evaluation"]["metrics"][
                "joint"
            ]
            for seed in SEEDS
        ]
        table[name]["oracle_removal"] = [
            summary[f"evidence-{seed}:native"]["oracle_evidence_removal"]
            for seed in SEEDS
        ]
        path = (
            ROOT / "data/train.jsonl"
            if name == "control"
            else ROOT / f"augmentation_data/{name}.jsonl"
        )
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        counts = {role: Counter() for role in ["before", "after", "other"]}
        for row in rows:
            for sp in row["spans"]:
                style = (
                    "만원"
                    if "만" in sp["text"]
                    else "천원" if "천" in sp["text"] else "원"
                )
                counts[sp["role"]][style] += 1
        diagnostics[name] = dict(amount_format_counts=counts, rows=len(rows))
    # Interactions are descriptive paired-seed effects in the constructed probe.
    interaction = {}
    for split in ["evaluation", "units", "distractor", "official"]:
        interaction[split] = {}
        for metric in ["axes", "roles", "joint"]:
            vals = {k: np.array(v[split][metric]["seeds"]) for k, v in table.items()}
            interaction[split][metric] = dict(
                units_only_minus_control=(vals["units"] - vals["control"]).tolist(),
                scope_only_minus_control=(vals["scope"] - vals["control"]).tolist(),
                both_minus_control=(vals["both"] - vals["control"]).tolist(),
                interaction=(
                    vals["both"] - vals["units"] - vals["scope"] + vals["control"]
                ).tolist(),
            )
    report = dict(
        table=table,
        training_audit=diagnostics,
        paired_effects=interaction,
        scope="Post-hoc controlled data intervention on already observed author-constructed probes; same target supervision and budgets; not a new blind test.",
        caveat="Excluded combinations remain excluded as TARGET labels. Distractor clauses may mention an excluded combination; no claim of zero textual exposure in augmentation.",
    )
    (ROOT / "augmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    rows = [
        "# 금액 표기·대상 문맥의 2×2 보강 진단\n",
        "v3.1에서 드러난 표면 단서를 확인한 뒤 설계한 **사후 실험**입니다. 평가 자료는 이미 관측했습니다. 같은 근거 학습 모델·960개 대상 사례·라벨·초기값·300회 갱신 시도 예산을 유지하고 학습 문장에만 개입했습니다. 3개 새 조건 × 3 seed = 9회 추가 학습입니다. 보강 없음은 이미 실행한 v3.1 조건을 그대로 재사용합니다.\n",
        "| 학습 자료 | 기본 공동 | 원 단위 변환 공동 | 다른 사업 추가 공동 | 공식 관계·상태 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, v in table.items():
        rows.append(
            f"| {NAMES[name]} | {v['evaluation']['joint']['mean']*100:.2f}% | {v['units']['joint']['mean']*100:.2f}% | {v['distractor']['joint']['mean']*100:.2f}% | {v['official']['axes']['mean']*12:.2f}/12 |"
        )
    rows += [
        "\n모두 세 초기값 평균입니다. 공식 열은 관계·상태만, 나머지는 금액 역할까지 모두 맞아야 하는 공동 일치입니다. **새 조합은 대상 정답 라벨에서 제외**했습니다. 보강의 다른 사업 문장에는 해당 관계·상태의 조합이 텍스트로 나타날 수 있으므로, 보강 결과에 대해 전체 입력에서 한 번도 보지 않은 조합이라고 주장하지 않습니다.\n",
        "## 발견한 표면 단서\n",
        "최초 학습 자료에서 이전 금액 960개는 모두 만원 표기였습니다. 원 단위로 바꾼 평가에서 파서의 값·위치는 보존됐지만 금액 역할이 무너졌습니다. 단위 보강에서는 이전·후속·기타 값이 각각 원·만원·천원 표기를 갖도록 했습니다. 관계·상태 문구와 대상 금액의 실제 값은 바꾸지 않았습니다.\n",
        "대상 문맥 보강은 다른 사업의 짧은 사건 설명과 두 금액을 앞·뒤에 나누어 배치합니다. 다른 사업의 관계·상태는 대상 라벨과 별도로 구성합니다. 이 두 개입을 나눠서 각 반례의 성능 변화를 봤습니다. 같은 합성 문법·인위적 사업명이라는 한계는 남습니다.\n",
        "## 분해 지표\n",
        "| 조건 | 변환 시 관계·상태 | 변환 시 금액 역할 | 방해 시 관계·상태 | 방해 시 금액 역할 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, v in table.items():
        rows.append(
            f"| {NAMES[name]} | {v['units']['axes']['mean']*100:.2f}% | {v['units']['roles']['mean']*100:.2f}% | {v['distractor']['axes']['mean']*100:.2f}% | {v['distractor']['roles']['mean']*100:.2f}% |"
        )
    rows += [
        "\n원 단위 파서는 정답 라벨 없이 입력만 읽습니다. 수치 제약은 모델이 골라낸 쌍이 이미 틀리면 이를 자동으로 복원하지 못합니다. [전체 seed·단위 분포·효과 분해](augmentation_report.json)와 각 결과 폴더의 원시 예측을 함께 보아야 합니다.\n",
        "## 해석과 다음 단계\n",
        "이 비교는 작성한 자료의 특정 단서에 대한 실패를 실제로 고칠 수 있는지 확인합니다. 자연 문장의 일반화, 근거의 인과적 충실성, 공식 구절의 임시 라벨 타당성은 별도입니다. 실제 공고에서 나타나는 화자·부정 범위·날짜·단위·표·문서 버전을 수집하고 독립 2인 판독으로 확인해야 합니다. 학습 결과를 본 뒤 공식 라벨을 바꾸거나 좋은 사례만 남기지 않았습니다.\n",
        "![2×2 학습 자료 보강](figures/training-data-interventions.svg)\n",
    ]
    (ROOT / "AUGMENTATION.md").write_text("\n".join(rows).rstrip() + "\n")
    font_manager.fontManager.addfont(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "svg.fonttype": "none",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3), constrained_layout=True)
    for ax, split, title in zip(
        axes,
        ["evaluation", "units", "distractor"],
        ["기본 합성 문장", "같은 금액의 원 단위 표기", "다른 사업 문장 덧붙이기"],
    ):
        for i, name in enumerate(NAMES):
            m = table[name][split]["joint"]
            ax.bar(
                i,
                m["mean"] * 100,
                color="#236247" if name == "both" else "#809986",
                width=0.65,
            )
            ax.scatter(
                [i] * 3, np.array(m["seeds"]) * 100, color="#ae682f", s=17, zorder=3
            )
            ax.text(
                i, m["max"] * 100 + 3, f"{m['mean']*100:.1f}", ha="center", fontsize=10
            )
        ax.set_xticks(range(4), ["없음", "단위", "문맥", "둘 다"])
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("학습 자료 보강")
        ax.set_ylabel("관계·상태·금액 공동 일치 (%)")
    fig.suptitle(
        "표기와 역할의 상관을 끊고, 다른 사업을 구별하도록 학습하기",
        fontsize=16,
        fontweight="bold",
    )
    fig.supxlabel(
        "동일 대상 960행 · 같은 모델·초기값·갱신 시도 예산 · 점: 각 seed · 사후 진단 / 새 독립 평가 아님",
        fontsize=10,
    )
    for ext in ["svg", "png", "pdf"]:
        path = ROOT / f"figures/training-data-interventions.{ext}"
        options = {"backend": "cairo"} if ext == "pdf" else {}
        fig.savefig(path, dpi=180, **options)
        if ext == "svg":
            path.write_text(
                "\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    for name, v in table.items():
        print(
            name,
            {
                k: round(v[k]["joint"]["mean"] * 100, 2)
                for k in ["evaluation", "units", "distractor"]
            },
            "official axes",
            round(v["official"]["axes"]["mean"] * 12, 2),
        )


if __name__ == "__main__":
    main()
