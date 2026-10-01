"""Authored development cases; labels and both references never enter requests."""

import json
from pathlib import Path
from logic import interpret

ROOT = Path(__file__).parent


def atom(f, op, v, e="E"):
    return dict(field=f, op=op, value=v, evidence=[e])


def prof(values, assertion=True, status="ready"):
    return dict(
        status=status,
        assertion=assertion,
        values=[dict(field=f, value=v) for f, v in values.items()],
    )


def rule(clauses):
    return dict(status="ready", clauses=clauses)


def main():
    bundles = []
    cases = []

    def add(key, scope, text, clauses, rows):
        ref = rule(clauses) if clauses else dict(status="insufficient", clauses=[])
        bundles.append(
            dict(
                id=key,
                scope=scope,
                evidence={"E": text},
                reference_policy=ref,
                origin="authored",
            )
        )
        for i, (claim, values, assertion, label) in enumerate(rows, 1):
            profile = prof(values, assertion)
            result = interpret(ref, profile, ["E"])["decision"]
            assert result == label, (key, i, result, label)
            cases.append(
                dict(
                    id=f"{key}-{i}",
                    bundle=key,
                    claim=claim,
                    reference_profile=profile,
                    label=label,
                )
            )

    A = atom
    add(
        "dev-age-income",
        "연령과 가구 중위소득 요건만",
        "만 19세 이상 39세 이하이고 가구 소득이 기준 중위소득 150% 이하인 사람을 대상으로 한다.",
        [
            [
                A("age_years", "ge", 19),
                A("age_years", "le", 39),
                A("income_pct", "le", 150),
            ]
        ],
        [
            (
                "만 39세이고 가구 소득이 중위소득 150%인 나는 이 두 요건을 충족한다.",
                {"age_years": 39, "income_pct": 150},
                True,
                "supported",
            ),
            (
                "만 40세인 나는 소득을 밝히지 않았어도 이 두 요건을 충족한다.",
                {"age_years": 40},
                True,
                "contradicted",
            ),
            (
                "만 25세이고 소득 정보가 없는 나는 이 두 요건을 충족하지 못한다.",
                {"age_years": 25},
                False,
                "not_established",
            ),
        ],
    )
    add(
        "dev-subject-or",
        "본인 또는 부모의 거주기간 요건만",
        "신청자 본인이 12개월 이상 거주했거나 부모가 36개월 이상 거주했다면 거주기간 요건을 충족한다.",
        [[A("residence_months", "ge", 12)], [A("parent_residence_months", "ge", 36)]],
        [
            (
                "나는 3개월, 부모는 3년 거주했다. 거주기간 요건을 충족한다.",
                {"residence_months": 3, "parent_residence_months": 36},
                True,
                "supported",
            ),
            (
                "부모는 1년 거주했고 내 거주기간은 밝히지 않았다. 거주기간 요건을 충족한다.",
                {"parent_residence_months": 12},
                True,
                "not_established",
            ),
            (
                "나는 11개월, 부모는 35개월 거주했다. 거주기간 요건을 충족하지 않는다.",
                {"residence_months": 11, "parent_residence_months": 35},
                False,
                "supported",
            ),
        ],
    )
    add(
        "dev-pledge",
        "실거주 또는 거주확약 요건만",
        "실거주 24개월 이상을 원칙으로 한다. 다만 24개월 미만이어도 향후 거주확약을 하면 이 요건을 충족한다.",
        [
            [A("residence_months", "ge", 24)],
            [A("residence_months", "lt", 24), A("pledge", "eq", True)],
        ],
        [
            (
                "나는 2년 거주했고 확약은 하지 않았다. 이 요건을 충족한다.",
                {"residence_months": 24, "pledge": False},
                True,
                "supported",
            ),
            (
                "나는 6개월 거주했고 거주확약을 했다. 이 요건을 충족한다.",
                {"residence_months": 6, "pledge": True},
                True,
                "supported",
            ),
            (
                "나는 거주확약을 했고 실제 거주기간은 밝히지 않았다. 이 요건을 충족한다.",
                {"pledge": True},
                True,
                "supported",
            ),
        ],
    )
    add(
        "dev-benefits",
        "주택보유 및 복지급여 제외요건만",
        "주택 소유자, 주거급여 또는 생계급여 수급자는 제외한다. 교육급여만 받는 사람은 이 사유로 제외하지 않는다.",
        [
            [
                A("homeowner", "eq", False),
                A("housing_benefit", "eq", False),
                A("livelihood_benefit", "eq", False),
            ]
        ],
        [
            (
                "무주택인 나는 교육급여만 받고 주거급여와 생계급여는 받지 않는다. 이 제외요건을 통과한다.",
                {
                    "homeowner": False,
                    "education_benefit": True,
                    "housing_benefit": False,
                    "livelihood_benefit": False,
                },
                True,
                "supported",
            ),
            (
                "나는 주택을 소유하고 있고 급여 수급 여부는 밝히지 않았다. 이 제외요건을 통과한다.",
                {"homeowner": True},
                True,
                "contradicted",
            ),
            (
                "무주택인 나는 주거급여는 받지 않고 생계급여 여부는 밝히지 않았다. 이 제외요건을 통과한다.",
                {"homeowner": False, "housing_benefit": False},
                True,
                "not_established",
            ),
        ],
    )
    add(
        "dev-foreign",
        "국적·영주권 제외 및 예외만",
        "외국 국적자는 제외한다. 단, 외국 국적자 중 영주권을 보유한 사람은 신청 가능하다.",
        [
            [A("foreign_national", "eq", False)],
            [A("foreign_national", "eq", True), A("permanent_resident", "eq", True)],
        ],
        [
            (
                "나는 외국 국적자이며 영주권이 있다. 이 국적 요건을 충족한다.",
                {"foreign_national": True, "permanent_resident": True},
                True,
                "supported",
            ),
            (
                "나는 외국 국적자이며 영주권이 없다. 이 국적 요건을 충족하지 못한다.",
                {"foreign_national": True, "permanent_resident": False},
                False,
                "supported",
            ),
            (
                "나는 국적을 밝히지 않았고 영주권은 없다. 이 국적 요건을 충족한다.",
                {"permanent_resident": False},
                True,
                "not_established",
            ),
        ],
    )
    add(
        "dev-education-work",
        "학적과 근로시간 요건만",
        "졸업·수료·중퇴자에 한한다. 이 중 미취업자 또는 주 20시간 미만 취업자에게 자격이 있다. 자영업자는 제외한다.",
        [
            [
                A("education_status", "in", ["graduated", "completed", "dropout"]),
                A("employment_status", "eq", "unemployed"),
            ],
            [
                A("education_status", "in", ["graduated", "completed", "dropout"]),
                A("employment_status", "eq", "employed"),
                A("weekly_work_hours", "lt", 20),
            ],
        ],
        [
            (
                "대학을 중퇴한 나는 취업하여 주 19시간 일한다. 이 요건을 충족한다.",
                {
                    "education_status": "dropout",
                    "employment_status": "employed",
                    "weekly_work_hours": 19,
                },
                True,
                "supported",
            ),
            (
                "대학을 졸업한 나는 취업하여 주 20시간 일한다. 이 요건을 충족한다.",
                {
                    "education_status": "graduated",
                    "employment_status": "employed",
                    "weekly_work_hours": 20,
                },
                True,
                "contradicted",
            ),
            (
                "대학 휴학 중인 나는 미취업자이다. 이 요건을 충족하지 못한다.",
                {"education_status": "leave", "employment_status": "unemployed"},
                False,
                "supported",
            ),
        ],
    )
    add(
        "dev-money",
        "임대보증금과 월세 상한 요건만",
        "보증금 8천만원 이하이면서 월세 60만원 이하인 임대차 계약을 지원한다.",
        [[A("deposit_krw", "le", 80000000), A("monthly_rent_krw", "le", 600000)]],
        [
            (
                "내 계약은 보증금 0.8억원, 월세 60만원이다. 두 상한을 충족한다.",
                {"deposit_krw": 80000000, "monthly_rent_krw": 600000},
                True,
                "supported",
            ),
            (
                "내 계약은 보증금 8천만원, 월세 61만원이다. 두 상한을 충족한다.",
                {"deposit_krw": 80000000, "monthly_rent_krw": 610000},
                True,
                "contradicted",
            ),
            (
                "내 월세는 50만원이고 보증금은 밝히지 않았다. 두 상한을 충족하지 못한다.",
                {"monthly_rent_krw": 500000},
                False,
                "not_established",
            ),
        ],
    )
    add(
        "dev-insufficient",
        "가구 소득 요건만",
        "가구 소득 기준은 별도 소득표를 따른다. 해당 표는 이 발췌문에 포함되어 있지 않다.",
        None,
        [
            (
                "내 가구 소득은 중위소득 120%이다. 소득 요건을 충족한다.",
                {"income_pct": 120},
                True,
                "not_established",
            ),
            (
                "내 가구 소득은 중위소득 180%이다. 소득 요건을 충족하지 못한다.",
                {"income_pct": 180},
                False,
                "not_established",
            ),
            (
                "내 연 소득은 3000만원이다. 소득 요건을 충족한다.",
                {"income_krw": 30000000},
                True,
                "not_established",
            ),
        ],
    )
    (ROOT / "development.json").write_text(
        json.dumps(dict(bundles=bundles, cases=cases), ensure_ascii=False, indent=2)
        + "\n"
    )
    print(len(bundles), len(cases))


if __name__ == "__main__":
    main()
