"""Render measured results without changing frozen data, training or analysis."""

import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from common import ROOT, datasets, save

NAMES = {
    "legacy_plain": "이전 모델",
    "legacy_marked": "이전 + 대상 표시",
    "plain": "질문 쌍 학습",
    "marked": "질문 쌍 + 대상 표시",
    "blind": "질문 가림 대조",
}
LAYOUTS = {
    "evaluation": "사업별 설명",
    "interleaved": "교차 배치",
    "anaphora": "지시어 배치",
    "order": "순서 반전",
    "units": "원 단위 표기",
    "absent": "대상 미등장",
}
REL = {
    "correction": "정정",
    "policy_change": "내용 변경",
    "restatement": "재표현",
    "unspecified": "관계 불명",
}
STATE = {
    "asserted": "확정 서술",
    "planned": "계획",
    "under_review": "검토·미확정",
    "denied": "부인",
    "hypothetical": "가정",
    "unspecified": "상태 불명",
}


def main():
    d = json.loads((ROOT / "summary.json").read_text())
    sets = datasets()
    pct = lambda x: f"{x*100:.2f}%"
    rows = [
        "# v5 관측 결과\n",
        "질문 대상의 의존성을 검사한 사후 연구입니다. 새로 작성한 유한 문법의 진단이며 실제 문서·독립 사람 정답 성능이 아닙니다.\n",
        "## 주지표: 같은 본문의 두 질문을 모두 맞히기\n",
        "| 조건 | 기본 행 공동 | 기본 두 질문 공동 | 교차 배치 두 질문 | 지시어 두 질문 | 순서 반전 두 질문 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, name in NAMES.items():
        t = d["table"][c]
        rows.append(
            f"| {name} | "
            + pct(t["evaluation"]["joint"]["mean"])
            + " | "
            + " | ".join(
                pct(t[s]["pair_joint"]["mean"])
                for s in ["evaluation", "interleaved", "anaphora", "order"]
            )
            + " |"
        )
    rows += [
        "\n세 초기값 평균. 기본·각 변형은 같은 192본문, 384질문입니다. 두 질문 공동은 각 질문의 관계·상태·모든 금액 역할이 함께 맞아야 합니다. 근거 F1은 별도입니다.\n",
        f"새 질문 쌍 학습에서 대상 표시를 추가했을 때 기본 두 질문 공동은 **{pct(d['table']['plain']['evaluation']['pair_joint']['mean'])} → {pct(d['table']['marked']['evaluation']['pair_joint']['mean'])}**였습니다. 교차 배치는 {pct(d['table']['plain']['interleaved']['pair_joint']['mean'])} → {pct(d['table']['marked']['interleaved']['pair_joint']['mean'])}, 지시어 배치는 {pct(d['table']['plain']['anaphora']['pair_joint']['mean'])} → {pct(d['table']['marked']['anaphora']['pair_joint']['mean'])}입니다. 사업명을 표시하는 처리가 담화 지시까지 해결했다고 해석하지 않습니다.\n",
        "질문 가림의 두 질문 공동은 입력에서 대상을 제거한 설계 때문에 0%여야 합니다. 의미 있는 비교는 대상을 제공한 두 학습 조건과 동일 가중치의 표시 전후이며, 가림 대조군 대비 차이를 기여의 크기로 내세우지 않습니다.\n",
        "## 무관한 값의 유입과 근거\n",
        "| 조건 | 기본 기타 오연결 | 교차 기타 오연결 | 지시어 기타 오연결 | 기본 근거 문자 F1 | 지시어 근거 문자 F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, name in NAMES.items():
        t = d["table"][c]
        cells = [
            f"{t[s]['other_wrong']}/{t[s]['other_total']} ({pct(t[s]['other_wrong']/t[s]['other_total'])})"
            for s in ["evaluation", "interleaved", "anaphora"]
        ]
        rows.append(
            f"| {name} | "
            + " | ".join(
                cells
                + [
                    pct(t[s]["evidence_character_f1"]["mean"])
                    for s in ["evaluation", "anaphora"]
                ]
            )
            + " |"
        )
    rows += [
        "\n오연결의 분모는 세 초기값을 합친 개별 금액 관측이며 본문 수가 아닙니다. 근거 F1은 대상의 관계·상태 문자 구절과의 일치로, 실제 설명의 충실성을 사람에게 평가받은 점수가 아닙니다.\n",
        "## 모든 초기값과 검증 문턱\n",
        "| 조건 | seed | 기본 두 질문 | 교차 두 질문 | 지시어 두 질문 | 문턱 | 교차 답변/오답 | 지시어 답변/오답 | 미등장 답변/오답 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csvrows = []
    for c, name in NAMES.items():
        for seed in [17, 42, 2026]:
            r = d["runs"][f"{c}/{seed}"]
            cells = [
                pct(r["sets"][s]["pairs"]["pair_joint"])
                for s in ["evaluation", "interleaved", "anaphora"]
            ]
            gates = [
                f"{r['sets'][s]['selective']['accepted']}/{r['sets'][s]['selective']['wrong']}"
                for s in ["interleaved", "anaphora", "absent"]
            ]
            rows.append(
                f"| {name} | {seed} | "
                + " | ".join(
                    cells
                    + [str(r["threshold"]) if r["threshold"] is not None else "없음"]
                    + gates
                )
                + " |"
            )
            for split, value in r["sets"].items():
                csvrows.append(
                    dict(
                        condition=c,
                        seed=seed,
                        split=split,
                        joint=value["metrics"]["joint"],
                        axes=value["metrics"]["axes"],
                        roles=value["metrics"]["roles"],
                        pair_joint=(
                            value["pairs"]["pair_joint"] if value["pairs"] else ""
                        ),
                        evidence_character_f1=value["metrics"]["evidence_character_f1"],
                        threshold=r["threshold"],
                        accepted=value["selective"]["accepted"],
                        wrong=value["selective"]["wrong"],
                    )
                )
    rows += [
        "\n보류는 검증 192행에서 선택했습니다. 문턱이 없는 조건은 답변하지 않으며, 이를 오류 0%의 성공으로 표현하지 않습니다. 미등장 사업은 모두 기타·불명이 정답인 새 진단입니다. 질문이 문자 그대로 없다는 이유로 자동 보류하는 별도 규칙은 이번 모델 결과에 더하지 않았습니다.\n",
        "## 고정 사례: 같은 본문, 두 질문\n",
        "아래 사례는 추론 전에 고정한 첫 기본 평가 본문, seed42입니다. 성공 여부로 고르지 않았습니다.\n",
    ]
    fixed = sets["evaluation"][:2]
    assert fixed[0]["document_id"] == "evaluation-8-0-0"
    example = {"rows": fixed, "seed": 42, "predictions": {}}
    rows.append("> " + fixed[0]["text"] + "\n")
    for c, name in NAMES.items():
        ps = json.loads((ROOT / "results" / c / "42/evaluation.json").read_text())[:2]
        example["predictions"][c] = ps
        rows.append(f"**{name}**\n")
        for r, p in zip(fixed, ps):
            before = (
                " · ".join(s["text"] for s in p["spans"] if s["role"] == "before")
                or "없음"
            )
            after = (
                " · ".join(s["text"] for s in p["spans"] if s["role"] == "after")
                or "없음"
            )
            rows.append(
                f"- {r['query']}: {REL[p['relation']]} / {STATE[p['state']]} · 기준 {before} / 후속 {after}"
            )
    save(ROOT / "fixed_example.json", example)
    routing = json.loads((ROOT / "routing_summary.json").read_text())
    rows += [
        "\n## 학습 자료에서도 남은 실패\n",
        f"대상 표시 모델의 학습 행 공동은 {pct(d['table']['marked']['train']['joint']['mean'])}, 학습 두 질문 공동은 {pct(d['table']['marked']['train']['pair_joint']['mean'])}였습니다. 평가 실패를 일반화 문제만으로 해석할 수 없습니다. 이번 상위 2개 층·300회 갱신 설정에서 학습 자료도 충분히 맞추지 못했습니다. 표시 기법이나 NLP의 근본 한계로 결론내리지 않습니다.\n",
        "## 결과를 본 뒤 추가한 단순 문장 선별 비교\n",
        "새 학습의 실패를 본 뒤 별도 설계·코드를 고정했습니다. 본문에서 질문 사업명이 들어 있는 문장만 골라 같은 v4 모델에 주었습니다. 원문 금액·근거 위치를 복원하고, 제외한 금액은 규칙으로 기타 처리합니다. 학습은 추가하지 않았습니다. 모델이 문장 경계를 학습했다고 주장하지 않습니다.\n",
        "| 배치 | 선별 전 두 질문 공동 | 선별 후 두 질문 공동 | 선별 가능 질문 비율 |",
        "|---|---:|---:|---:|",
    ]
    for split in ["evaluation", "interleaved", "anaphora", "order"]:
        t = routing["table"][split]
        rows.append(
            f"| {LAYOUTS[split]} | {pct(d['table']['legacy_plain'][split]['pair_joint']['mean'])} | {pct(t['pair_joint']['mean'])} | {pct(t['retrieval_coverage']['mean'])} |"
        )
    rows += [
        "\n선별되지 않은 질문을 정답으로 세지 않았습니다. 위 공동 지표의 분모는 여전히 전체 192본문입니다. 지시어 배치에서 이름은 소개 문장에만 있어 금액을 찾지 못하면 보류합니다. 이는 지시어 판독 성공이 아닙니다. 대상 미등장도 명시적 보류이며 정답률 성공으로 합산하지 않습니다.\n",
        "| seed | 기본 두 질문 | 교차 두 질문 | 문턱 | 기본 답변/오답 | 교차 답변/오답 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed in [17, 42, 2026]:
        rr = routing["runs"][str(seed)]
        cells = [
            pct(rr["sets"][s]["pair_joint"]) for s in ["evaluation", "interleaved"]
        ]
        gates = [
            f"{rr['sets'][s]['selective']['accepted']}/{rr['sets'][s]['selective']['wrong']}"
            for s in ["evaluation", "interleaved"]
        ]
        rows.append(
            f"| {seed} | " + " | ".join(cells + [str(rr["threshold"])] + gates) + " |"
        )
    rows += [
        "\n이 비교는 결과를 본 뒤 추가한 사후 탐색입니다. 이전 모델에 대한 문장 선별의 효과를 보지만, 자연 문장·화자·부분 이름·한 문장 안의 복수 사업까지 해결한 증거는 아닙니다. 전체 [선별 설계](routing_protocol.json), [잠금](routing_freeze.json), [원예측](routing_results/), [결과](routing_summary.json), [검증](routing_verification.json)을 공개합니다.\n"
    ]
    rows += [
        "\n## 무엇을 확인했고, 무엇은 남았는가\n",
        "원 단위 변형은 모든 조건에서 통일된 입력이 같으므로 확률도 정확히 같아야 합니다. 질문 가림 쌍도 같은 입력이므로 확률이 같아야 합니다. 이는 결정적 입력 처리와 검증의 성질이며 수치·지시어 이해의 학습을 증명하지 않습니다.\n",
        "기본 family별 결과, 전체 학습·검증 결과, 유형별 답변율과 원확률은 [summary.json](summary.json)과 [results/](results/)에 있습니다. 일부 family나 초기값만 골라 평균을 만들지 않았습니다. 신뢰구간을 384개 독립 언어 표본인 것처럼 계산하지 않았습니다.\n",
        "실제 문서와 화자·부정 범위 판독은 이번 결과로 입증하지 않았습니다. 정확 문자열로 대상을 알려 주는 비용·별칭 실패, 값 정보 손실, 유한 문법과 AI 설계 라벨이라는 제약이 남습니다. 다음 비교는 지시어·별칭을 포함한 대상 근거 연결과 실제 사건 단위 독립 판독이어야 합니다.\n",
        "[설계와 재현 안내](README.md) · [검증 기록](verification.json) · [연구 기록](research_record.json)\n",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(rows))
    with (ROOT / "results-table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0]))
        w.writeheader()
        w.writerows(csvrows)
    plot(d, routing)


def plot(d, routing):
    font_manager.fontManager.addfont(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "policy-binding-v5",
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(
        "질문을 바꾸면 연결한 금액과 근거도 바뀌는가? · v5",
        fontsize=19,
        fontweight="bold",
    )
    c = list(NAMES)
    y = [d["table"][k]["evaluation"]["pair_joint"]["mean"] * 100 for k in c]
    y.append(routing["table"]["evaluation"]["pair_joint"]["mean"] * 100)
    ax = axes[0, 0]
    ax.barh(
        range(6),
        y,
        color=["#9aafb1", "#66898d", "#83a6e0", "#0b7370", "#c4c4c4", "#b8743f"],
    )
    ax.set_yticks(range(6), list(NAMES.values()) + ["사후 문장 선별"])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_title("A. 기본 본문: 두 질문 공동 일치 (%)", loc="left")
    for i, v in enumerate(y):
        ax.text(v + 1, i, f"{v:.2f}", va="center")
    splits = ["evaluation", "interleaved", "anaphora"]
    x = np.arange(3)
    ax = axes[0, 1]
    for k, color in [
        ("legacy_plain", "#8c9799"),
        ("plain", "#5485c4"),
        ("marked", "#0b7370"),
    ]:
        yy = [d["table"][k][s]["pair_joint"]["mean"] * 100 for s in splits]
        ax.plot(x, yy, "o-", label=NAMES[k], color=color)
    ax.set_xticks(x, [LAYOUTS[s] for s in splits])
    ax.plot(
        x,
        [routing["table"][s]["pair_joint"]["mean"] * 100 for s in splits],
        "s--",
        label="사후 문장 선별",
        color="#b8743f",
    )
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=9)
    ax.set_title("B. 같은 사건, 다른 배치: 두 질문 공동 (%)", loc="left")
    ax = axes[1, 0]
    for i, k in enumerate(["plain", "marked"]):
        yy = [
            d["table"][k][s]["other_wrong"] / d["table"][k][s]["other_total"] * 100
            for s in splits
        ]
        bars = ax.bar(
            x + (i - 0.5) * 0.32,
            yy,
            0.32,
            label=NAMES[k],
            color=["#5485c4", "#0b7370"][i],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    ax.set_xticks(x, [LAYOUTS[s] for s in splits])
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.set_title("C. 기타 금액을 대상에 잘못 연결한 비율 (%)", loc="left")
    ax = axes[1, 1]
    for i, key in enumerate(["joint", "pair_joint"]):
        yy = [d["table"]["marked"][s][key]["mean"] * 100 for s in splits]
        bars = ax.bar(
            x + (i - 0.5) * 0.32,
            yy,
            0.32,
            label=["행 공동: 384질문", "두 질문 공동: 192본문"][i],
            color=["#87bcb1", "#0b7370"][i],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    ax.set_xticks(x, [LAYOUTS[s] for s in splits])
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.set_title("D. 대상 표시 모델: 서로 다른 분모 (%)", loc="left")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel(
        "세 초기값 평균 · 유한 문법의 AI 작성 진단 · 변형은 같은 평가 사건 · 독립 사람 판독 0건 · 선별 실패는 보류이며 공동 정답에서 제외",
        fontsize=10,
    )
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "query-binding.png", dpi=170)
    fig.savefig(out / "query-binding.svg", metadata={"Date": None})
    fig.savefig(out / "query-binding.pdf", backend="cairo")
    plt.close(fig)
    svg = out / "query-binding.svg"
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
