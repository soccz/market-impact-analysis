"""Authored claims and scoped reference programs for newly collected notices."""

import json
from pathlib import Path
from build_development import atom, prof, rule
from logic import interpret

ROOT = Path(__file__).parent


def main():
    bundles = []
    cases = []
    spans = json.loads((ROOT / "source_spans.json").read_text())

    def A(f, op, v, e):
        return atom(f, op, v, e)

    def add(key, scope, ids, clauses, rows):
        ref = rule(clauses)
        bundles.append(
            dict(
                id=key,
                scope=scope,
                evidence={i: spans[i] for i in ids},
                reference_policy=ref,
                origin="new_document_scoped",
            )
        )
        for i, (claim, values, assertion, label) in enumerate(rows, 1):
            q = prof(values, assertion)
            assert interpret(ref, q, ids)["decision"] == label, (key, i)
            cases.append(
                dict(
                    id=f"{key}-{i}",
                    bundle=key,
                    claim=claim,
                    reference_profile=q,
                    label=label,
                )
            )

    db = [
        A(f, "eq", False, "D-BENEFITS")
        for f in ["livelihood_benefit", "medical_benefit", "housing_benefit"]
    ]
    dc = A("residence_city", "eq", "대전광역시", "D-CITY")
    di = A("income_pct", "le", 150, "D-INCOME")
    add(
        "dj-city-income",
        "신청일의 주민등록 도시와 공고의 건강보험 기준으로 이미 산정된 가구 중위소득 비율 요건만. 주택·나이·증빙·선정순위는 범위 밖이다.",
        ["D-CITY", "D-INCOME"],
        [[dc, di]],
        [
            (
                "신청일에 내 주민등록은 대전광역시이고 공고 방식으로 산정한 가구 중위소득 비율은 150%이다. 이 두 요건을 충족한다.",
                {"residence_city": "대전광역시", "income_pct": 150},
                True,
                "supported",
            ),
            (
                "신청일에 내 주민등록은 세종특별자치시이고 가구 중위소득 비율은 100%이다. 이 두 요건을 충족한다.",
                {"residence_city": "세종특별자치시", "income_pct": 100},
                True,
                "contradicted",
            ),
            (
                "부모의 주민등록은 대전광역시이다. 내 주민등록 도시는 밝히지 않았고 내 가구 중위소득 비율은 140%이다. 나는 이 두 요건을 충족한다.",
                {"income_pct": 140},
                True,
                "not_established",
            ),
            (
                "신청일에 내 주민등록은 대전광역시이고 가구 중위소득 비율은 151%이다. 이 두 요건을 충족하지 못한다.",
                {"residence_city": "대전광역시", "income_pct": 151},
                False,
                "supported",
            ),
        ],
    )
    add(
        "dj-benefits",
        "공고의 기초생활 급여 수급에 따른 제외 요건만. 통과한다는 말은 이 제외 사유가 없다는 뜻이다.",
        ["D-BENEFITS"],
        [db],
        [
            (
                "나는 의료급여만 받고 생계급여와 주거급여는 받지 않는다. 이 급여 제외요건을 통과한다.",
                {
                    "medical_benefit": True,
                    "livelihood_benefit": False,
                    "housing_benefit": False,
                },
                True,
                "contradicted",
            ),
            (
                "나는 교육급여만 받고 생계·의료·주거급여는 받지 않는다. 이 급여 제외요건을 통과한다.",
                {
                    "education_benefit": True,
                    "livelihood_benefit": False,
                    "medical_benefit": False,
                    "housing_benefit": False,
                },
                True,
                "supported",
            ),
            (
                "나는 생계급여와 주거급여를 받지 않지만 의료급여 여부는 밝히지 않았다. 이 급여 제외요건을 통과한다.",
                {"livelihood_benefit": False, "housing_benefit": False},
                True,
                "not_established",
            ),
            (
                "나는 생계·의료·주거급여를 모두 받지 않는다. 이 급여 제외요건을 통과하지 못한다.",
                {
                    "livelihood_benefit": False,
                    "medical_benefit": False,
                    "housing_benefit": False,
                },
                False,
                "contradicted",
            ),
        ],
    )
    add(
        "dj-city-benefits",
        "신청일 주민등록 도시 요건과 기초생활 급여 수급 제외 요건의 동시 충족만. 나머지 요건은 범위 밖이다.",
        ["D-CITY", "D-BENEFITS"],
        [[dc] + db],
        [
            (
                "내 주민등록은 신청일에 대전광역시이며 교육급여만 받고 생계·의료·주거급여는 받지 않는다. 이 두 종류의 요건을 충족한다.",
                {
                    "residence_city": "대전광역시",
                    "education_benefit": True,
                    "livelihood_benefit": False,
                    "medical_benefit": False,
                    "housing_benefit": False,
                },
                True,
                "supported",
            ),
            (
                "내 주민등록은 신청일에 대전광역시이며 생계급여를 받고 다른 급여 여부는 밝히지 않았다. 이 두 종류의 요건을 충족한다.",
                {"residence_city": "대전광역시", "livelihood_benefit": True},
                True,
                "contradicted",
            ),
            (
                "신청일 내 주민등록 도시는 밝히지 않았다. 생계·의료·주거급여는 받지 않는다. 이 두 종류의 요건을 충족한다.",
                {
                    "livelihood_benefit": False,
                    "medical_benefit": False,
                    "housing_benefit": False,
                },
                True,
                "not_established",
            ),
            (
                "내 주민등록은 신청일에 부산광역시이고 급여 여부는 밝히지 않았다. 이 두 종류의 요건을 충족하지 못한다.",
                {"residence_city": "부산광역시"},
                False,
                "supported",
            ),
        ],
    )
    add(
        "dj-foreign",
        "외국 국적자 제외 여부만. 재외국민 여부와 다른 요건은 평가 범위 밖이다.",
        ["D-FOREIGN"],
        [[A("foreign_national", "eq", False, "D-FOREIGN")]],
        [
            (
                "나는 외국 국적자이고 대한민국 영주권이 있다. 이 외국 국적 제외요건을 통과한다.",
                {"foreign_national": True, "permanent_resident": True},
                True,
                "contradicted",
            ),
            (
                "나는 외국 국적자가 아니며 대한민국 국적자이다. 이 외국 국적 제외요건을 통과한다.",
                {"foreign_national": False},
                True,
                "supported",
            ),
            (
                "나는 영주권이 없으며 국적은 밝히지 않았다. 이 외국 국적 제외요건을 통과한다.",
                {"permanent_resident": False},
                True,
                "not_established",
            ),
            (
                "나는 외국 국적자이다. 이 외국 국적 제외요건을 통과하지 못한다.",
                {"foreign_national": True},
                False,
                "supported",
            ),
        ],
    )
    gc = A("residence_city", "eq", "광주광역시", "G-CITY")
    ga = [A("age_years", "ge", 19, "G-AGE"), A("age_years", "le", 39, "G-AGE")]
    gi = A("income_pct", "le", 150, "G-INCOME")
    add(
        "gj-city-age-income",
        "공고일인 2025-02-05의 주민등록 도시·만 나이·공고가 지정한 2024년 건강보험 기준으로 이미 산정한 가구 중위소득 비율만. 학력·근로·다른 요건은 범위 밖이다.",
        ["G-CITY", "G-AGE", "G-INCOME"],
        [[gc] + ga + [gi]],
        [
            (
                "공고일에 내 주민등록은 광주광역시, 만 나이는 39세이고 지정한 방식으로 산정한 가구 중위소득 비율은 150%이다. 이 세 요건을 충족한다.",
                {"residence_city": "광주광역시", "age_years": 39, "income_pct": 150},
                True,
                "supported",
            ),
            (
                "공고일에 내 주민등록은 광주광역시, 만 나이는 18세이고 가구 중위소득 비율은 100%이다. 이 세 요건을 충족한다.",
                {"residence_city": "광주광역시", "age_years": 18, "income_pct": 100},
                True,
                "contradicted",
            ),
            (
                "공고일에 내 주민등록은 광주광역시이고 만 25세이다. 가구 소득 정보는 없다. 이 세 요건을 충족하지 못한다.",
                {"residence_city": "광주광역시", "age_years": 25},
                False,
                "not_established",
            ),
            (
                "공고일에 내 주민등록은 경기도 광주시이며 만 30세이고 가구 중위소득 비율은 120%이다. 이 세 요건을 충족하지 못한다.",
                {"residence_city": "경기도 광주시", "age_years": 30, "income_pct": 120},
                False,
                "supported",
            ),
        ],
    )
    add(
        "gj-benefits",
        "공고의 생계급여·조건부 생계급여 제외 및 의료·교육·주거급여 예외만. livelihood_benefit은 생계 또는 조건부 생계급여 수급을 포함한다.",
        ["G-BENEFITS"],
        [[A("livelihood_benefit", "eq", False, "G-BENEFITS")]],
        [
            (
                "나는 의료급여만 받고 생계급여와 주거급여는 받지 않는다. 이 급여 제외요건을 통과한다.",
                {
                    "medical_benefit": True,
                    "livelihood_benefit": False,
                    "housing_benefit": False,
                },
                True,
                "supported",
            ),
            (
                "나는 주거급여와 생계급여를 함께 받는다. 이 급여 제외요건을 통과한다.",
                {"housing_benefit": True, "livelihood_benefit": True},
                True,
                "contradicted",
            ),
            (
                "나는 의료급여를 받지 않고 생계급여 여부는 밝히지 않았다. 이 급여 제외요건을 통과한다.",
                {"medical_benefit": False},
                True,
                "not_established",
            ),
            (
                "나는 조건부 생계급여를 받고 있다. 이 급여 제외요건을 통과하지 못한다.",
                {"livelihood_benefit": True},
                False,
                "supported",
            ),
        ],
    )
    gw = [
        [A("employment_status", "eq", "unemployed", "G-UNEMPLOYED")],
        [
            A("employment_status", "eq", "employed", "G-HOURS"),
            A("weekly_work_hours", "lt", 30, "G-HOURS"),
        ],
    ]
    workscope = "미취업 또는 단기 상용근로의 주당 시간 요건만. 사업자등록·휴폐업·공무원 합격·증빙은 범위 밖이다. unemployed는 명시된 미취업/고용보험 미가입, employed는 명시된 고용보험 가입 상용근로 상태에 대응한다."
    add(
        "gj-work",
        workscope,
        ["G-UNEMPLOYED", "G-HOURS"],
        gw,
        [
            (
                "나는 고용보험에 가입한 단기 상용근로자이고 주 29시간 근무한다. 이 근로시간 요건을 충족한다.",
                {"employment_status": "employed", "weekly_work_hours": 29},
                True,
                "supported",
            ),
            (
                "나는 고용보험에 가입한 상용근로자이고 주 30시간 근무한다. 이 근로시간 요건을 충족한다.",
                {"employment_status": "employed", "weekly_work_hours": 30},
                True,
                "contradicted",
            ),
            (
                "나는 고용보험에 가입한 상용근로자이지만 주당 근로시간은 밝히지 않았다. 이 근로시간 요건을 충족한다.",
                {"employment_status": "employed"},
                True,
                "not_established",
            ),
            (
                "나는 미취업자이며 고용보험에 가입되어 있지 않다. 이 근로시간 요건을 충족하지 못한다.",
                {"employment_status": "unemployed"},
                False,
                "contradicted",
            ),
        ],
    )
    ge = A(
        "education_status", "in", ["graduated", "completed", "dropout"], "G-EDUCATION"
    )
    add(
        "gj-education-work",
        "일반 국내 학교의 학력 요건과 "
        + workscope
        + " 원격대학·학점은행제 재학 및 졸업예정자 예외는 이 범위에서 제외한다. enrolled/leave는 예외에 해당하지 않는 일반 학교의 재학/휴학이며 제적 여부는 이번 주장에 없다.",
        ["G-EDUCATION", "G-EDUCATION-EXCEPTIONS", "G-UNEMPLOYED", "G-HOURS"],
        [[ge] + c for c in gw],
        [
            (
                "나는 일반 대학을 중퇴했고 고용보험에 가입한 단기 상용근로자로 주 29시간 일한다. 이 학력·근로 요건을 충족한다.",
                {
                    "education_status": "dropout",
                    "employment_status": "employed",
                    "weekly_work_hours": 29,
                },
                True,
                "supported",
            ),
            (
                "나는 일반 대학 휴학생이며 미취업자로 고용보험에 가입되어 있지 않다. 이 학력·근로 요건을 충족한다.",
                {"education_status": "leave", "employment_status": "unemployed"},
                True,
                "contradicted",
            ),
            (
                "나는 일반 대학을 수료했고 고용보험에 가입한 상용근로자이나 주당 근로시간은 밝히지 않았다. 이 학력·근로 요건을 충족한다.",
                {"education_status": "completed", "employment_status": "employed"},
                True,
                "not_established",
            ),
            (
                "나는 일반 대학 졸업자이고 고용보험에 가입한 상용근로자로 주 30시간 일한다. 이 학력·근로 요건을 충족하지 못한다.",
                {
                    "education_status": "graduated",
                    "employment_status": "employed",
                    "weekly_work_hours": 30,
                },
                False,
                "supported",
            ),
        ],
    )
    (ROOT / "official_reference.json").write_text(
        json.dumps(dict(bundles=bundles, cases=cases), ensure_ascii=False, indent=2)
        + "\n"
    )
    print(len(bundles), len(cases))


if __name__ == "__main__":
    main()
