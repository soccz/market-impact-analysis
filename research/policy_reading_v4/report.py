"""Export complete results and a visual account of gains and scope errors."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from common import ROOT

NAMES = {
    "raw": "원문",
    "raw_scope": "원문 + 문맥 보강",
    "normalized": "표기 통일",
    "normalized_scope": "표기 통일 + 문맥 보강",
}
SHORT = ["원문", "원문\n+ 문맥", "표기 통일", "표기 통일\n+ 문맥"]


def main():
    d = json.loads((ROOT / "summary.json").read_text())
    errors = json.loads((ROOT / "error_groups.json").read_text())

    def pct(x):
        return f"{x*100:.2f}%"

    def joint(condition, split):
        return pct(d["table"][condition]["native"][split]["joint"]["mean"])

    nm = errors["matrices"]["normalized_scope"]

    rows = [
        "# v4 관측 결과\n",
        "v3 실패 이후 설계한 사후 비교입니다. 새 학습 6회와 기존 모델 6개의 재추론을 구분합니다. 합성 문법·선정된 공식 구절·AI 임시 라벨이며 사람 검수 전입니다.\n",
        "## 공동 일치: 관계·상태·모든 금액 역할\n",
        "| 조건 | 기본 | 원 단위 | 다른 사업 | 두 반례 동시 | 공식 관계·상태 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, name in NAMES.items():
        v = d["table"][c]["native"]
        rows.append(
            f"| {name} | "
            + " | ".join(
                pct(v[s]["joint"]["mean"])
                for s in ["evaluation", "units", "distractor", "combined"]
            )
            + f" | {v['official']['axes']['mean']*12:.2f}/12 |"
        )
    rows += [
        "\n세 초기값 평균입니다. 공식 열은 관계·상태만이며 나머지 공동 지표와 합산하지 않습니다. 공식 문자 기준선은 v3에서 4/12였습니다. 같은 선정 구절에 대한 비교이며 독립 실제 문서 시험이 아닙니다.\n",
        "## 개선과 역효과\n",
        f"표기 통일만 적용한 기본·원 단위 공동 일치는 각각 {joint('normalized','evaluation')} / {joint('normalized','units')}입니다. 같은 값의 표기를 같은 입력으로 만든 결과로, 두 조건의 예측 확률도 같습니다. 이는 수치 의미를 새로 학습했다는 증거가 아니며, 같은 오답을 일관되게 낼 수도 있습니다.\n",
        f"문맥 보강 조건에 표기 통일을 추가한 변화는 원 단위 {joint('raw_scope','units')} → {joint('normalized_scope','units')}, 다른 사업 {joint('raw_scope','distractor')} → {joint('normalized_scope','distractor')}, 두 반례 동시 {joint('raw_scope','combined')} → {joint('normalized_scope','combined')}입니다. 한 종류의 반례 개선으로 다른 반례나 자연 문장의 판독 성능까지 설명할 수 없습니다.\n",
        "| 문맥 보강 조건 | 다른 사업 반례의 관계·상태 | 모든 금액 역할 일치 | 기타 금액을 기준·후속으로 오연결 |",
        "|---|---:|---:|---:|",
    ]
    for c in ["raw_scope", "normalized_scope"]:
        m = errors["matrices"][c][2]
        false = sum(m[:2]) / sum(m)
        v = d["table"][c]["native"]["distractor"]
        rows.append(
            f"| {NAMES[c]} | {pct(v['axes']['mean'])} | {pct(v['roles']['mean'])} | {pct(false)} ({sum(m[:2])}/{sum(m)}) |"
        )
    rows += [
        f"\n오연결은 세 초기값의 개별 금액 관측을 합친 비율이며, 행 전체의 모든 역할 일치와 분모가 다릅니다. 표기 통일 + 문맥 보강의 대상 기준 역할은 {nm[0][0]:,}/{sum(nm[0]):,}, 후속 역할은 {nm[1][1]:,}/{sum(nm[1]):,}개를 맞혔습니다. 기타 금액 {sum(nm[2]):,}개 중 {sum(nm[2][:2]):,}개를 대상 역할로 엮었습니다. **대상 값의 누락과 무관한 값의 연결을 분리해 보아야 합니다.** 오류 집계는 결과를 본 뒤의 기술적 분석이며 추가 학습하지 않았습니다.\n",
        "## 모든 초기값과 보류\n",
        "| 조건 | seed | 기본 공동 | 원 단위 공동 | 다른 사업 공동 | 복합 공동 | 검증 문턱 | 복합 답변 / 오답 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c, name in NAMES.items():
        for seed in [17, 42, 2026]:
            r = d["runs"][f"{c}/{seed}:native"]
            g = r["sets"]["combined"]["selective"]
            rows.append(
                f"| {name} | {seed} | "
                + " | ".join(
                    pct(r["sets"][s]["metrics"]["joint"])
                    for s in ["evaluation", "units", "distractor", "combined"]
                )
                + f" | {r['threshold'] if r['threshold'] is not None else '없음'} | {g['accepted']} / {g['wrong']} |"
            )
    rows += [
        "\n보류 문턱은 원래 검증 160행에서 선택했습니다. 새 반례 분포의 오류를 보장하지 않습니다. 답변 0개는 판독 능력의 증명이 아니며, 낮은 답변 오류와 충분한 유형별 답변율을 함께 보아야 합니다. [전체 유형별 답변·오류](summary.json)에 공식 구절까지 포함했습니다.\n",
        "## 수치 제약의 별도 비교\n",
        "| 조건 | 기본 공동 | 원 단위 공동 | 다른 사업 공동 | 복합 공동 |",
        "|---|---:|---:|---:|---:|",
    ]
    for c, name in NAMES.items():
        v = d["table"][c]["numeric"]
        rows.append(
            f"| {name} | "
            + " | ".join(
                pct(v[s]["joint"]["mean"])
                for s in ["evaluation", "units", "distractor", "combined"]
            )
            + " |"
        )
    b = d["paired_frame_bootstrap"]
    rows += [
        "\n모델이 선택한 금액 역할을 사용한 제약이며 정답 역할을 넣지 않았습니다. 무관한 금액을 연결하는 오류나 화자·부정 범위를 자동으로 복원하지 못합니다.\n",
        "## 제한된 쌍 비교와 정보 손실\n",
        f"미리 정한 복합 반례 비교의 차이는 {b['mean']*100:.2f}%p, frame 묶음 bootstrap 구간은 {b['interval95'][0]*100:.2f}–{b['interval95'][1]*100:.2f}%p입니다. 단 6개의 합성 frame 안에서 초기값 평균의 차이를 재표집한 값이며, 자연 문장 모집단이나 사람 판독 불확실성의 신뢰구간이 아닙니다.\n",
        "표기 통일은 다른 실제 값까지 같은 문맥 표현으로 만듭니다. [정보 손실 예시](data/value_information.json)에서 동일 금액과 다른 금액이 같은 모델 입력에 대응합니다. 값·원문 위치를 별도로 보관해도 인코더 자체의 크기·방향·비교 추론 능력이 복원되는 것은 아닙니다.\n",
        "## 다음 검증 질문\n",
        "다음 후보는 금액마다 사업·사건·화자와 연결되는 범위를 명시하는 판독입니다. 예컨대 금액 주변 표기를 지우더라도 대상 사업의 금액만 남기는지, 다른 기관의 부인 문장이 들어와도 대상의 상태가 유지되는지 비교해야 합니다. 현재 결과는 이 구조를 아직 구현·검증한 것이 아닙니다. 새 실제 공고 버전과 독립 2인 판독도 별도로 필요합니다.\n",
        "![표기 안정성과 범위 오류](figures/amount-view-tradeoff.svg)\n",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(rows).rstrip() + "\n")
    font_manager.fontManager.addfont(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    x = np.arange(4)
    for ax, splits, title in [
        (axs[0, 0], ["evaluation", "units"], "A. 표기 변화에 대한 안정성"),
        (axs[0, 1], ["distractor", "combined"], "B. 다른 사업과 복합 반례"),
    ]:
        for i, (split, label, color) in enumerate(
            zip(
                splits,
                [
                    "기본" if splits[0] == "evaluation" else "다른 사업 문장",
                    "원 단위" if splits[0] == "evaluation" else "문장 + 원 단위",
                ],
                ["#88a596", "#276348"],
            )
        ):
            vals = [
                d["table"][c]["native"][split]["joint"]["mean"] * 100 for c in NAMES
            ]
            bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, label=label, color=color)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_xticks(x, SHORT)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("관계·상태·모든 금액 공동 일치 (%)")
        ax.set_title(title, loc="left", fontweight="bold", pad=30)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    ax = axs[1, 0]
    left = np.zeros(2)
    matrix = np.array(
        [errors["matrices"][c][2] for c in ["raw_scope", "normalized_scope"]]
    )
    matrix = matrix / matrix.sum(1, keepdims=True) * 100
    for i, (label, color) in enumerate(
        zip(
            ["기준으로 오연결", "후속으로 오연결", "기타로 유지"],
            ["#bd7c4d", "#d5b374", "#276348"],
        )
    ):
        ax.barh([0, 1], matrix[:, i], left=left, label=label, color=color)
        for y, v in enumerate(matrix[:, i]):
            if v > 8:
                ax.text(
                    left[y] + v / 2,
                    y,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if i == 2 else "#262b27",
                )
        left += matrix[:, i]
    ax.set_yticks([0, 1], ["원문 + 문맥", "표기 통일 + 문맥"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("정답이 기타인 금액 3,888개 관측 · 세 seed 합산 (%)")
    ax.set_title(
        "C. 무관한 금액을 대상 값으로 엮는 오류", loc="left", fontweight="bold", pad=30
    )
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1), ncol=3, fontsize=9, frameon=False
    )
    ax = axs[1, 1]
    for i, c in enumerate(NAMES):
        vals = np.array(d["table"][c]["native"]["official"]["axes"]["values"]) * 12
        ax.scatter(np.full(3, i), vals, color="#276348", zorder=3)
        ax.hlines(vals.mean(), i - 0.2, i + 0.2, color="#bb7437", linewidth=2)
    ax.axhline(4, linestyle="--", color="#7b837d", label="v3 문자 기준선 4/12")
    ax.set_xticks(x, SHORT)
    ax.set_ylim(0, 12.5)
    ax.set_ylabel("관계·상태 동시 일치 수 / 12")
    ax.set_title("D. 공식 구절의 관계·상태 일치", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    fig.suptitle(
        "금액 표기 통일이 보존한 것과 남긴 오류", fontweight="bold", fontsize=17
    )
    fig.supxlabel(
        "사후 합성 진단 · 동일 모델/초기값/갱신 시도 · 공식 구절은 AI 임시 판독 / 사람 검수 전",
        fontsize=10,
    )
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for ext in ["svg", "png", "pdf"]:
        path = out / f"amount-view-tradeoff.{ext}"
        fig.savefig(path, dpi=175, **({"backend": "cairo"} if ext == "pdf" else {}))
        if ext == "svg":
            path.write_text(
                "\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    print("Wrote complete results and SVG/PNG/PDF figure")


if __name__ == "__main__":
    main()
