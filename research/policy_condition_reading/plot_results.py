"""Export a descriptive count plot. No confidence or generalization claim."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_cairo import FigureCanvasCairo

ROOT = Path(__file__).resolve().parent
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
summary = json.loads((ROOT / "results/summary.json").read_text())
labels = [
    "전체 문서 첫 일치",
    "문자 검색 1개",
    "KLUE 평균 표현 1개",
    "문자 검색 3개 일치",
]
fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
for i, (method, name) in enumerate(
    zip(["document_first", "tfidf_top1", "klue_top1", "tfidf_consensus3"], labels)
):
    counts = summary["metrics"][method]
    left = 0
    for key, color in [
        ("correct_joint", "#245b55"),
        ("wrong_value", "#b45d43"),
        ("abstain", "#ded9ce"),
    ]:
        value = counts[key]
        ax.barh(i, value, left=left, color=color, height=0.56)
        if value:
            ax.text(
                left + value / 2,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if key != "abstain" else "#383a37",
            )
        left += value
ax.set_yticks(range(4), labels, fontproperties=font)
ax.invert_yaxis()
ax.set_xlim(0, 30)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("30개 잠정 조건: 값+근거 일치 / 잘못된 값 / 보류", fontproperties=font)
ax.set_title(
    "같은 원문·같은 파서, 근거를 찾는 방법의 차이",
    fontproperties=font,
    fontsize=15,
    pad=18,
)
for edge in ["top", "right", "left"]:
    ax.spines[edge].set_visible(False)
fig.text(
    0.5,
    0.025,
    "4개 사업·8개 공고의 탐색 · 독립 사람 검수 0명 · 별도 미확인 2항목은 모두 보류",
    ha="center",
    fontproperties=font,
    fontsize=10,
)
out = ROOT / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "condition-reading.png", dpi=180, bbox_inches="tight")
FigureCanvasCairo(fig).print_pdf(str(out / "condition-reading.pdf"))
print("Exported PNG and PDF")
