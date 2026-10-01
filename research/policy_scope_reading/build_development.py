"""Author development probes; neither real policy nor independent human labels."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    rows = []

    def add(group, sentences, claims):
        for i, (claim, label, support) in enumerate(claims, 1):
            rows.append(
                dict(
                    id=f"dev-{group}-{i}",
                    split="development",
                    family=group,
                    source_kind="authored",
                    sentences={f"S{j}": s for j, s in enumerate(sentences, 1)},
                    claim=claim,
                    reference=dict(decision=label, evidence=[f"S{j}" for j in support]),
                )
            )

    add(
        "recipient",
        [
            "첫 신청자는 기준중위소득 150% 이하이어야 한다.",
            "2024년 수혜자는 기준중위소득 120% 이하이어야 한다.",
        ],
        [
            (
                "2024년 수혜자이며 소득 140%인 가상 신청자는 이 소득 조건을 충족한다.",
                "contradicted",
                [2],
            ),
            (
                "수혜 이력이 없는 첫 신청자이며 소득 140%인 가상 신청자는 이 소득 조건을 충족한다.",
                "supported",
                [1],
            ),
        ],
    )
    add(
        "priority",
        [
            "지원 자격은 기준중위소득 150% 이하이다.",
            "신청자가 많으면 소득 85% 이하인 사람을 우선 선발한다.",
        ],
        [
            ("소득 100%인 가상 신청자는 이 소득 자격을 충족한다.", "supported", [1]),
            (
                "우선 선발 조건은 모든 신청자에게 적용되는 소득 자격 상한이 85%임을 뜻한다.",
                "contradicted",
                [1, 2],
            ),
        ],
    )
    add(
        "possibility",
        [
            "다른 기관 장학금 신청 예정자는 제외 대상이 될 수 있으므로 신청 전 확인해야 한다.",
            "댐 주변 학생 장학금도 확인할 예시 중 하나이다.",
        ],
        [
            (
                "다른 기관 장학금을 신청할 예정이라는 사실만으로 반드시 제외된다고 확정할 수 있다.",
                "not_established",
                [1],
            ),
            (
                "댐 주변 학생 장학금 신청 예정자는 확인할 필요가 없다.",
                "contradicted",
                [1, 2],
            ),
        ],
    )
    add(
        "negation",
        [
            "지원금을 30만원으로 줄인다는 주장은 사실이 아니다.",
            "현행 지원액은 50만원이다.",
        ],
        [
            ("현행 지원액은 30만원이다.", "contradicted", [1, 2]),
            ("지원금을 줄인다는 발표가 확정되었다.", "contradicted", [1]),
        ],
    )
    add(
        "exception",
        [
            "기초생활급여 수급자는 신청할 수 없다.",
            "다만 교육급여만 받는 경우는 신청할 수 있다.",
        ],
        [
            (
                "교육급여만 받는 가상 신청자는 이 급여 제외 조건에 걸리지 않는다.",
                "supported",
                [1, 2],
            ),
            (
                "생계급여와 교육급여를 함께 받는 가상 신청자는 이 급여 제외 조건에 걸리지 않는다.",
                "contradicted",
                [1, 2],
            ),
        ],
    )
    add(
        "version",
        [
            "수정 공고는 9월 16일 이후 신청 건에 적용한다.",
            "그 전에 접수된 신청 건에는 이전 공고를 적용한다.",
            "수정 공고의 소득 상한은 150%이고 이전 공고는 120%이다.",
        ],
        [
            (
                "9월 10일 접수했고 소득 140%인 가상 신청자는 적용되는 소득 상한을 충족한다.",
                "contradicted",
                [2, 3],
            ),
            (
                "9월 20일 접수했고 소득 140%인 가상 신청자는 적용되는 소득 상한을 충족한다.",
                "supported",
                [1, 3],
            ),
        ],
    )
    add(
        "alternative",
        [
            "매출 요건은 전년도 매출 1,200만원 이상 또는 최근 3개월 평균 월 매출 100만원 이상이다."
        ],
        [
            (
                "전년도 900만원이고 최근 3개월 평균 월 110만원인 가상 신청자는 이 매출 요건을 충족한다.",
                "supported",
                [1],
            ),
            (
                "전년도 900만원이고 최근 3개월 평균 월 90만원인 가상 신청자는 이 매출 요건을 충족한다.",
                "contradicted",
                [1],
            ),
        ],
    )
    add(
        "conjunction",
        ["대상은 거주 기간 6개월 이상이면서 만 19세 이상인 사람이다."],
        [
            (
                "8개월 거주한 만 18세 가상 신청자는 이 두 조건을 모두 충족한다.",
                "contradicted",
                [1],
            ),
            (
                "8개월 거주한 만 20세 가상 신청자는 이 두 조건을 모두 충족한다.",
                "supported",
                [1],
            ),
        ],
    )
    add(
        "missing",
        ["소득 세부 기준은 별표 2에 따른다.", "이 입력에는 별표 2가 제공되지 않았다."],
        [
            (
                "가상 신청자에게 적용되는 소득 상한은 150%이다.",
                "not_established",
                [1, 2],
            ),
            (
                "이 입력의 본문만으로 소득 상한을 확정할 수 있다.",
                "contradicted",
                [1, 2],
            ),
        ],
    )
    add(
        "amount_role",
        [
            "사업 전체 예산은 5억원이다.",
            "지원액은 신청자 1인당 최대 30만원이며 실제 지급은 인정 비용에 따른다.",
        ],
        [
            ("신청자 1인당 지원 상한은 5억원이다.", "contradicted", [1, 2]),
            ("가상 신청자의 실제 지급액은 정확히 30만원이다.", "not_established", [2]),
        ],
    )
    add(
        "quoted_old",
        [
            "작년 안내에는 만 34세 이하라고 기재되어 있었다.",
            "이번 모집의 연령 상한은 만 39세이다.",
        ],
        [
            (
                "만 36세인 가상 신청자는 이번 모집의 연령 상한을 충족한다.",
                "supported",
                [2],
            ),
            ("이번 모집의 연령 상한은 만 34세이다.", "contradicted", [1, 2]),
        ],
    )
    add(
        "other_program",
        [
            "가온사업은 소득 120% 이하를 신청 자격으로 한다.",
            "다른 사업인 누리사업의 자격은 소득 150% 이하이다.",
        ],
        [
            (
                "가온사업에 신청하는 소득 140%인 가상 신청자는 이 소득 조건을 충족한다.",
                "contradicted",
                [1],
            ),
            (
                "누리사업에 신청하는 소득 140%인 가상 신청자는 이 소득 조건을 충족한다.",
                "supported",
                [2],
            ),
        ],
    )
    return rows


if __name__ == "__main__":
    (ROOT / "development.json").write_text(
        json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    )
