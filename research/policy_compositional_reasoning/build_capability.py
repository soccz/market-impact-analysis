"""Post-collection expansion into arithmetic and previously excluded school exceptions."""

import json
from pathlib import Path
from build_development import atom as A, prof
from extended_logic import interpret

ROOT = Path(__file__).parent


def main():
    spans = json.loads((ROOT / "source_spans.json").read_text())
    bs = []
    cs = []

    def add(key, scope, ids, program, rows):
        bs.append(
            dict(
                id=key,
                scope=scope,
                evidence={i: spans[i] for i in ids},
                reference_policy=program,
                origin="post_collection_capability",
            )
        )
        for i, (claim, vals, pol, label) in enumerate(rows, 1):
            q = prof(vals, pol)
            assert interpret(program, q, ids)["decision"] == label, (key, i)
            cs.append(
                dict(
                    id=f"{key}-{i}",
                    bundle=key,
                    claim=claim,
                    reference_profile=q,
                    label=label,
                )
            )

    def d(f, op, v):
        return A(f, op, v, "D-RENT")

    rent = dict(
        status="ready",
        derived=[
            dict(
                field="derived_0",
                constant=0,
                terms=[
                    dict(field="monthly_rent_krw", numerator=1, denominator=1),
                    dict(field="deposit_krw", numerator=0.05, denominator=12),
                ],
                evidence=["D-RENT"],
            )
        ],
        clauses=[
            [d("deposit_krw", "le", 100000000), d("monthly_rent_krw", "le", 600000)],
            [d("monthly_rent_krw", "gt", 600000), d("derived_0", "le", 800000)],
        ],
    )
    add(
        "cap-rent",
        "공고 거주요건의 보증금·월세 상한 및 월세 초과 시 환산액 예외만. 다른 자격·증빙·건물유형·공동임차 분담은 범위 밖이며 질문 금액은 이미 본인 부담액이다.",
        ["D-RENT"],
        rent,
        [
            (
                "내 보증금은 1억원, 월세는 60만원이다. 이 금액 요건을 충족한다.",
                {"deposit_krw": 100000000, "monthly_rent_krw": 600000},
                True,
                "supported",
            ),
            (
                "내 보증금은 2천만원, 월세는 70만원이다. 이 금액 요건을 충족한다.",
                {"deposit_krw": 20000000, "monthly_rent_krw": 700000},
                True,
                "supported",
            ),
            (
                "내 보증금은 2400만원, 월세는 70만원이다. 이 금액 요건을 충족한다.",
                {"deposit_krw": 24000000, "monthly_rent_krw": 700000},
                True,
                "supported",
            ),
            (
                "내 보증금은 2400만 240원, 월세는 70만원이다. 이 금액 요건을 충족한다.",
                {"deposit_krw": 24000240, "monthly_rent_krw": 700000},
                True,
                "contradicted",
            ),
            (
                "내 보증금은 2천만원, 월세는 75만원이다. 이 금액 요건을 충족하지 못한다.",
                {"deposit_krw": 20000000, "monthly_rent_krw": 750000},
                False,
                "supported",
            ),
            (
                "내 보증금은 밝히지 않았고 월세는 70만원이다. 이 금액 요건을 충족한다.",
                {"monthly_rent_krw": 700000},
                True,
                "not_established",
            ),
            (
                "내 보증금은 0.8억원, 월세는 60만원이다. 이 금액 요건을 충족하지 못한다.",
                {"deposit_krw": 80000000, "monthly_rent_krw": 600000},
                False,
                "contradicted",
            ),
            (
                "내 보증금은 1억 1천만원, 월세는 60만원이다. 이 금액 요건을 충족한다.",
                {"deposit_krw": 110000000, "monthly_rent_krw": 600000},
                True,
                "contradicted",
            ),
        ],
    )

    def e(f, op, v, source="G-EDUCATION-EXCEPTIONS"):
        return A(f, op, v, source)

    ed = dict(
        status="ready",
        derived=[],
        clauses=[
            [
                e(
                    "education_status",
                    "in",
                    ["graduated", "completed", "dropout", "expelled"],
                    "G-EDUCATION",
                )
            ],
            [
                e("education_status", "eq", "graduation_expected"),
                e("expected_graduation_month", "eq", "2025-02"),
                e("completion_certificate", "eq", True),
            ],
            [
                e("education_status", "eq", "enrolled"),
                e("school_kind", "in", ["remote", "credit_bank"]),
                e("prior_graduation", "eq", True),
            ],
        ],
    )
    add(
        "cap-education",
        "국내 학교의 학력 요건과 원격대학·학점은행제 재학 및 2025년 2월 졸업예정자 예외만. 이전 학교 졸업과 현재 과정 학적을 구분한다. 해외 번역·다른 자격은 범위 밖이며 필요한 졸업증명서 제출 여부는 별도 요건으로 평가하지 않는다.",
        ["G-EDUCATION", "G-EDUCATION-EXCEPTIONS"],
        ed,
        [
            (
                "나는 현재 사이버대학에 재학 중이고 이전에 고등학교를 졸업했다. 이 학력 요건을 충족한다.",
                {
                    "education_status": "enrolled",
                    "school_kind": "remote",
                    "prior_graduation": True,
                },
                True,
                "supported",
            ),
            (
                "나는 현재 일반 국내 대학에 재학 중이고 이전에 고등학교를 졸업했다. 졸업예정자는 아니다. 이 학력 요건을 충족한다.",
                {
                    "education_status": "enrolled",
                    "school_kind": "general",
                    "prior_graduation": True,
                },
                True,
                "contradicted",
            ),
            (
                "나는 현재 학점은행제 재학생이며 이전 학교 졸업 사실은 밝히지 않았다. 이 학력 요건을 충족한다.",
                {"education_status": "enrolled", "school_kind": "credit_bank"},
                True,
                "not_established",
            ),
            (
                "나는 일반 국내 대학에서 2025년 2월 졸업예정자이며 현재 과정의 수료증명서를 발급받을 수 있다. 이 학력 요건을 충족한다.",
                {
                    "education_status": "graduation_expected",
                    "school_kind": "general",
                    "expected_graduation_month": "2025-02",
                    "completion_certificate": True,
                },
                True,
                "supported",
            ),
            (
                "나는 일반 국내 대학에서 2025년 2월 졸업예정자이며 현재 과정의 수료증명서를 발급받을 수 없다. 이 학력 요건을 충족한다.",
                {
                    "education_status": "graduation_expected",
                    "school_kind": "general",
                    "expected_graduation_month": "2025-02",
                    "completion_certificate": False,
                },
                True,
                "contradicted",
            ),
            (
                "나는 현재 방송통신대학 재학생이며 이전 고교·대학·대학원 졸업 사실은 없다. 이 학력 요건을 충족한다.",
                {
                    "education_status": "enrolled",
                    "school_kind": "remote",
                    "prior_graduation": False,
                },
                True,
                "contradicted",
            ),
            (
                "나는 일반 국내 대학을 졸업했다. 현재 재학 중인 학교는 없다. 이 학력 요건을 충족한다.",
                {"education_status": "graduated", "school_kind": "general"},
                True,
                "supported",
            ),
            (
                "나는 현재 사이버대학 재학생이며 이전에 고등학교를 졸업했다. 이 학력 요건을 충족하지 못한다.",
                {
                    "education_status": "enrolled",
                    "school_kind": "remote",
                    "prior_graduation": True,
                },
                False,
                "contradicted",
            ),
        ],
    )
    (ROOT / "capability_reference.json").write_text(
        json.dumps(dict(bundles=bs, cases=cs), ensure_ascii=False, indent=2) + "\n"
    )
    print(len(bs), len(cs))


if __name__ == "__main__":
    main()
