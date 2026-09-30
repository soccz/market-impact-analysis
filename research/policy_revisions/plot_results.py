"""Export figures from stored measurements; no manually entered model scores."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent


def main():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    enc = json.loads((ROOT / "results/encoder/results.json").read_text())["sets"]
    char = json.loads((ROOT / "results/character/results.json").read_text())["sets"]["evaluation"]["metrics"]
    audit = json.loads((ROOT / "verification.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1.05, 1]})
    labels = ["문자 관계·근거", "문맥 관계 + 등장 순서", "문맥 관계 + 토큰 근거"]
    values = [char["span_model"]["joint_exact"], enc["evaluation"]["learned_order"]["joint_exact"], enc["evaluation"]["span_model"]["joint_exact"]]
    axes[0].barh(range(3), [v * 100 for v in values], color=["#a8b39b", "#839d73", "#426537"], height=.5)
    axes[0].set_yticks(range(3), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("관계·금액 역할·위치·값 공동 일치 (%)")
    axes[0].set_title("A. 근거를 연결하면 무엇이 달라지는가", loc="left", pad=24)
    for i, v in enumerate(values):
        display = (Decimal(str(v)) * 100).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
        axes[0].text(v * 100 + 2, i, f"{display}%", va="center")
    classes = audit["results"]["encoder"]["evaluation"]["selective"]["by_true_relation"]
    names = ["correction", "policy_change", "equivalent", "undetermined"]
    answers = [classes[c]["accepted"] for c in names]
    ratios = [classes[c]["accepted"] / classes[c]["n"] * 100 for c in names]
    axes[1].barh(range(4), ratios, color="#426537", height=.5, label="답함")
    axes[1].barh(range(4), [100-v for v in ratios], left=ratios, color="#dedfd6", height=.5, label="보류")
    axes[1].set_yticks(range(4), ["오기 정정", "정책 변경", "같은 금액", "관계 불명"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("유형별 답변율과 보류율 (%)")
    axes[1].set_title("B. 오류 0건 뒤에 숨은 유형 누락", loc="left", pad=24)
    for i, n in enumerate(answers):
        axes[1].text(98, i, f"{n}/192 답변", ha="right", va="center", color="#343a2e")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
    fig.suptitle("한국어 정책 관계·금액 근거 판독 — 실제 실행한 사후 탐색", fontsize=15, y=.98)
    fig.text(.02, .055, "합성 평가 768개 = 숫자·사업명 정규화 뒤 40개 표현. 문맥 모델은 고정 KLUE-RoBERTa + 선형 분류기.\n오류 0건은 132개 응답에 한정되며 정정·정책 변경에는 0개 응답. 공식 구절 5개에서는 채택한 1개가 오답.\nAI 구성 정답·선정 공식 사례의 임시 판독이며, 독립 사람 평가와 실제 정책 일반화 검증 전.", fontsize=9, linespacing=1.7)
    fig.subplots_adjust(left=.16, right=.98, top=.79, bottom=.32, wspace=.62)
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ["svg", "png", "pdf"]:
        fig.savefig(folder / f"evidence-and-coverage.{extension}", dpi=180, facecolor="#fffef9")
    svg = folder / "evidence-and-coverage.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print("Exported SVG, PNG and PDF from measured results.")


if __name__ == "__main__":
    main()
