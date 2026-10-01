"""New authored combinations frozen after the logic-example method, before calls."""

import json
from pathlib import Path
from build_development import atom as A, prof, rule
from logic import interpret

ROOT = Path(__file__).parent


def main():
    bs = []
    cs = []

    def add(key, scope, text, clauses, rows):
        p = rule(clauses) if clauses else {"status": "insufficient", "clauses": []}
        bs.append(
            dict(
                id=key,
                scope=scope,
                evidence={"E": text},
                reference_policy=p,
                origin="authored_new_composition",
            )
        )
        for i, (claim, vals, pol, label) in enumerate(rows, 1):
            q = prof(vals, pol)
            assert interpret(p, q, ["E"])["decision"] == label, (key, i)
            cs.append(
                dict(
                    id=f"{key}-{i}",
                    bundle=key,
                    claim=claim,
                    reference_profile=q,
                    label=label,
                )
            )

    add(
        "mix-age-or-income",
        "아래 가상 사업의 나이·소득·거주기간 요건만",
        "만 23세 이상이고, 가구 중위소득 130% 이하이거나 본인이 18개월 이상 거주한 사람에게 자격이 있다. 나이 조건은 어느 경로에서나 충족해야 한다.",
        [
            [A("age_years", "ge", 23), A("income_pct", "le", 130)],
            [A("age_years", "ge", 23), A("residence_months", "ge", 18)],
        ],
        [
            (
                "나는 만 23세, 가구 중위소득 180%이며 본인 거주기간은 1년 6개월이다. 이 요건을 충족한다.",
                {"age_years": 23, "income_pct": 180, "residence_months": 18},
                True,
                "supported",
            ),
            (
                "나는 만 22세이며 가구 중위소득 100%, 본인 거주기간 24개월이다. 이 요건을 충족한다.",
                {"age_years": 22, "income_pct": 100, "residence_months": 24},
                True,
                "contradicted",
            ),
            (
                "나는 만 30세이고 가구 중위소득 131%이다. 부모는 24개월 거주했으며 내 거주기간은 밝히지 않았다. 이 요건을 충족한다.",
                {"age_years": 30, "income_pct": 131, "parent_residence_months": 24},
                True,
                "not_established",
            ),
            (
                "나는 만 25세이고 가구 중위소득은 130%이다. 거주기간 정보는 없다. 이 요건을 충족하지 못한다.",
                {"age_years": 25, "income_pct": 130},
                False,
                "contradicted",
            ),
        ],
    )
    add(
        "mix-exception-benefit",
        "국적 예외와 생계급여 제외의 동시 충족만",
        "생계급여 수급자는 제외한다. 외국 국적자는 원칙적으로 제외하지만 영주권이 있으면 국적 사유로 제외하지 않는다. 영주권 예외도 생계급여 제한을 면제하지는 않는다.",
        [
            [A("livelihood_benefit", "eq", False), A("foreign_national", "eq", False)],
            [
                A("livelihood_benefit", "eq", False),
                A("foreign_national", "eq", True),
                A("permanent_resident", "eq", True),
            ],
        ],
        [
            (
                "나는 외국 국적자이고 영주권이 있으며 생계급여를 받지 않는다. 이 두 제외요건을 통과한다.",
                {
                    "foreign_national": True,
                    "permanent_resident": True,
                    "livelihood_benefit": False,
                },
                True,
                "supported",
            ),
            (
                "나는 외국 국적자이고 영주권이 있지만 생계급여를 받는다. 이 두 제외요건을 통과한다.",
                {
                    "foreign_national": True,
                    "permanent_resident": True,
                    "livelihood_benefit": True,
                },
                True,
                "contradicted",
            ),
            (
                "나는 외국 국적자이고 영주권이 있다. 생계급여 여부는 밝히지 않았다. 이 두 제외요건을 통과한다.",
                {"foreign_national": True, "permanent_resident": True},
                True,
                "not_established",
            ),
            (
                "나는 외국 국적자가 아니고 영주권이 없으며 생계급여도 받지 않는다. 이 두 제외요건을 통과하지 못한다.",
                {
                    "foreign_national": False,
                    "permanent_resident": False,
                    "livelihood_benefit": False,
                },
                False,
                "contradicted",
            ),
        ],
    )
    add(
        "mix-date-money",
        "가입일에 따라 달라지는 임대보증금 상한만",
        "2025년 6월 1일 이전 가입자는 임대보증금 7천만원 이하, 당일 또는 이후 가입자는 9천만원 이하이어야 한다.",
        [
            [A("joined_on", "lt", "2025-06-01"), A("deposit_krw", "le", 70000000)],
            [A("joined_on", "ge", "2025-06-01"), A("deposit_krw", "le", 90000000)],
        ],
        [
            (
                "나는 2025년 6월 1일에 가입했고 보증금은 0.9억원이다. 이 상한을 충족한다.",
                {"joined_on": "2025-06-01", "deposit_krw": 90000000},
                True,
                "supported",
            ),
            (
                "나는 2025년 5월 31일에 가입했고 보증금은 8천만원이다. 이 상한을 충족한다.",
                {"joined_on": "2025-05-31", "deposit_krw": 80000000},
                True,
                "contradicted",
            ),
            (
                "나는 2025년 6월 2일에 신청했고 가입일은 밝히지 않았다. 보증금은 8천만원이다. 이 상한을 충족한다.",
                {"applied_on": "2025-06-02", "deposit_krw": 80000000},
                True,
                "not_established",
            ),
            (
                "나는 가입일을 밝히지 않았고 보증금은 7천만원이다. 이 상한을 충족한다.",
                {"deposit_krw": 70000000},
                True,
                "supported",
            ),
        ],
    )
    add(
        "mix-parent-or-pledge",
        "가상 거주·확약·주택 요건만",
        "무주택자 중에서 부모가 30개월 이상 거주했거나 본인이 거주확약을 한 사람에게 자격이 있다.",
        [
            [A("homeowner", "eq", False), A("parent_residence_months", "ge", 30)],
            [A("homeowner", "eq", False), A("pledge", "eq", True)],
        ],
        [
            (
                "나는 무주택이고 부모의 거주기간은 2년 6개월이며 거주확약은 하지 않았다. 이 요건을 충족한다.",
                {"homeowner": False, "parent_residence_months": 30, "pledge": False},
                True,
                "supported",
            ),
            (
                "나는 주택 소유자이며 거주확약을 했다. 부모의 거주기간은 밝히지 않았다. 이 요건을 충족한다.",
                {"homeowner": True, "pledge": True},
                True,
                "contradicted",
            ),
            (
                "나는 무주택이며 본인은 30개월 거주했고 거주확약은 하지 않았다. 부모의 거주기간은 밝히지 않았다. 이 요건을 충족한다.",
                {"homeowner": False, "residence_months": 30, "pledge": False},
                True,
                "not_established",
            ),
            (
                "나는 무주택이며 거주확약을 했다. 부모의 거주기간은 밝히지 않았다. 이 요건을 충족하지 못한다.",
                {"homeowner": False, "pledge": True},
                False,
                "contradicted",
            ),
        ],
    )
    add(
        "mix-age-partition",
        "가상 문안의 나이 분기와 소득 조건만. 논리 진단용으로 의도적으로 겹치는 결론을 썼다.",
        "만 27세 미만이면서 가구 중위소득 125% 이하인 사람, 또는 만 27세 이상이면서 가구 중위소득 125% 이하인 사람을 대상으로 한다.",
        [
            [A("age_years", "lt", 27), A("income_pct", "le", 125)],
            [A("age_years", "ge", 27), A("income_pct", "le", 125)],
        ],
        [
            (
                "내 나이는 밝히지 않았고 가구 중위소득은 125%이다. 이 요건을 충족한다.",
                {"income_pct": 125},
                True,
                "supported",
            ),
            (
                "내 나이는 밝히지 않았고 가구 중위소득은 126%이다. 이 요건을 충족한다.",
                {"income_pct": 126},
                True,
                "contradicted",
            ),
            (
                "나는 만 27세이며 가구 소득을 밝히지 않았다. 이 요건을 충족한다.",
                {"age_years": 27},
                True,
                "not_established",
            ),
            (
                "내 나이는 밝히지 않았고 가구 중위소득은 100%이다. 이 요건을 충족하지 못한다.",
                {"income_pct": 100},
                False,
                "contradicted",
            ),
        ],
    )
    add(
        "mix-impossible",
        "가상 문안에 명시된 소득 조건만. 모순된 초안에 대한 논리 진단이다.",
        "같은 가구 소득이 기준 중위소득 140%를 초과하면서 동시에 120% 이하여야 한다.",
        [[A("income_pct", "gt", 140), A("income_pct", "le", 120)]],
        [
            (
                "내 가구 소득 정보는 없다. 이 두 조건을 동시에 충족한다.",
                {},
                True,
                "contradicted",
            ),
            (
                "내 가구 중위소득은 130%이다. 이 두 조건을 동시에 충족한다.",
                {"income_pct": 130},
                True,
                "contradicted",
            ),
            (
                "내 가구 소득 정보는 없다. 이 두 조건을 동시에 충족하지 못한다.",
                {},
                False,
                "supported",
            ),
            (
                "내 가구 중위소득은 120%이다. 이 두 조건을 동시에 충족하지 못한다.",
                {"income_pct": 120},
                False,
                "supported",
            ),
        ],
    )
    add(
        "mix-education-or",
        "가상 학력·근로 두 경로만",
        "수료자는 미취업 여부와 관계없이 가능하다. 졸업자는 미취업자이거나 주 16시간 미만 임금근로자인 경우 가능하다. 이 밖의 학적은 대상이 아니다.",
        [
            [A("education_status", "eq", "completed")],
            [
                A("education_status", "eq", "graduated"),
                A("employment_status", "eq", "unemployed"),
            ],
            [
                A("education_status", "eq", "graduated"),
                A("employment_status", "eq", "employed"),
                A("weekly_work_hours", "lt", 16),
            ],
        ],
        [
            (
                "나는 수료자이며 주 40시간 임금근로를 한다. 이 요건을 충족한다.",
                {
                    "education_status": "completed",
                    "employment_status": "employed",
                    "weekly_work_hours": 40,
                },
                True,
                "supported",
            ),
            (
                "나는 졸업자이며 주 16시간 임금근로를 한다. 이 요건을 충족한다.",
                {
                    "education_status": "graduated",
                    "employment_status": "employed",
                    "weekly_work_hours": 16,
                },
                True,
                "contradicted",
            ),
            (
                "나는 졸업자이며 임금근로를 하지만 주당 시간은 밝히지 않았다. 이 요건을 충족한다.",
                {"education_status": "graduated", "employment_status": "employed"},
                True,
                "not_established",
            ),
            (
                "나는 수료자이며 취업 상태를 밝히지 않았다. 이 요건을 충족하지 못한다.",
                {"education_status": "completed"},
                False,
                "contradicted",
            ),
        ],
    )
    add(
        "mix-unavailable",
        "가상 임대보증금 상한 요건만",
        "임대보증금 상한은 미제공 별표에 따르며 이 문장에는 그 액수가 없다.",
        None,
        [
            (
                "내 보증금은 500만원이다. 이 상한을 충족한다.",
                {"deposit_krw": 5000000},
                True,
                "not_established",
            ),
            (
                "내 보증금은 2억원이다. 이 상한을 충족하지 못한다.",
                {"deposit_krw": 200000000},
                False,
                "not_established",
            ),
            (
                "내 월세는 45만원이며 보증금은 밝히지 않았다. 이 상한을 충족한다.",
                {"monthly_rent_krw": 450000},
                True,
                "not_established",
            ),
            (
                "내 보증금은 0원이다. 이 상한을 충족한다.",
                {"deposit_krw": 0},
                True,
                "not_established",
            ),
        ],
    )
    (ROOT / "composition.json").write_text(
        json.dumps(dict(bundles=bs, cases=cs), ensure_ascii=False, indent=2) + "\n"
    )
    print(len(bs), len(cs))


if __name__ == "__main__":
    main()
