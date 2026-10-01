"""Authored paired controls: preserve notation versus change meaning, not a population."""

import json
from bridge import ROOT, interpret


def atom(f, op, v):
    return dict(field=f, op=op, value=v, evidence=["E"])


def policy(clauses):
    return dict(status="ready", clauses=clauses, derived=[])


def profile(v, pol=True):
    return dict(
        status="ready",
        assertion=pol,
        values=[dict(field=f, value=x) for f, x in v.items()],
    )


def build():
    bs = []
    cs = []

    def group(name, evidence, p, scope, rows):
        bs.append(
            dict(
                id=name,
                scope=scope,
                evidence={"E": evidence},
                reference_policy=p,
                origin="authored",
            )
        )
        for v, text, facts, pol in rows:
            q = profile(facts, pol)
            label = interpret(p, q, ["E"])["decision"]
            cs.append(
                dict(
                    id=name + "-" + v,
                    bundle=name,
                    group=name,
                    variant=v,
                    claim=text,
                    reference_profile=q,
                    label=label,
                )
            )

    group(
        "mixed-money",
        "신청자 보증금이 24,000,000원 이하이면 이 조건을 충족한다.",
        policy([[atom("deposit_krw", "le", 24000000)]]),
        "신청자 본인의 현재 임대차보증금 상한만.",
        [
            (
                "base",
                "내 보증금은 2400만 240원이다. 이 요건을 충족한다.",
                {"deposit_krw": 24000240},
                True,
            ),
            (
                "preserve",
                "내 보증금은 24,000,240원이다. 이 요건을 충족한다.",
                {"deposit_krw": 24000240},
                True,
            ),
            (
                "change",
                "내 보증금은 2400만원이다. 이 요건을 충족한다.",
                {"deposit_krw": 24000000},
                True,
            ),
            (
                "unknown",
                "부모의 보증금은 2400만 240원이고 내 보증금은 밝히지 않았다. 이 요건을 충족한다.",
                {},
                True,
            ),
        ],
    )
    group(
        "decimal-money",
        "신청자 월세가 750,000원 미만이면 이 조건을 충족한다.",
        policy([[atom("monthly_rent_krw", "lt", 750000)]]),
        "신청자 월세 상한만. 보증금·관리비는 범위 밖.",
        [
            (
                "base",
                "내 월세는 75만원이다. 이 요건을 충족한다.",
                {"monthly_rent_krw": 750000},
                True,
            ),
            (
                "preserve",
                "내 월세는 0.0075억원이다. 이 요건을 충족한다.",
                {"monthly_rent_krw": 750000},
                True,
            ),
            (
                "change",
                "내 월세는 74만 9999원이다. 이 요건을 충족한다.",
                {"monthly_rent_krw": 749999},
                True,
            ),
            ("unknown", "내 월세는 밝히지 않았다. 이 요건을 충족한다.", {}, True),
        ],
    )
    group(
        "money-polarity",
        "신청자 보증금은 90,000,000원 이하이고 월세는 700,000원 이하여야 한다.",
        policy(
            [
                [
                    atom("deposit_krw", "le", 90000000),
                    atom("monthly_rent_krw", "le", 700000),
                ]
            ]
        ),
        "신청자 본인 보증금·월세의 두 상한만.",
        [
            (
                "base",
                "내 보증금은 0.9억원, 월세는 70만원이다. 이 요건을 충족한다.",
                {"deposit_krw": 90000000, "monthly_rent_krw": 700000},
                True,
            ),
            (
                "preserve",
                "내 보증금은 9천만원, 월세는 700,000원이다. 이 요건을 충족한다.",
                {"deposit_krw": 90000000, "monthly_rent_krw": 700000},
                True,
            ),
            (
                "change",
                "내 보증금은 0.9억원, 월세는 70만원이다. 이 요건을 충족하지 못한다.",
                {"deposit_krw": 90000000, "monthly_rent_krw": 700000},
                False,
            ),
            (
                "unknown",
                "내 보증금은 0.9억원, 월세는 밝히지 않았다. 이 요건을 충족한다.",
                {"deposit_krw": 90000000},
                True,
            ),
        ],
    )
    group(
        "subject",
        "신청자 본인이 해당 지역에 24개월 이상 연속 거주해야 한다.",
        policy([[atom("residence_months", "ge", 24)]]),
        "신청자 본인의 연속 거주기간만.",
        [
            (
                "base",
                "나는 24개월 연속 거주했다. 이 요건을 충족한다.",
                {"residence_months": 24},
                True,
            ),
            (
                "preserve",
                "나는 2년간 연속 거주했다. 이 요건을 충족한다.",
                {"residence_months": 24},
                True,
            ),
            (
                "change",
                "나는 23개월 연속 거주했다. 이 요건을 충족한다.",
                {"residence_months": 23},
                True,
            ),
            (
                "unknown",
                "부모는 24개월 연속 거주했고 내 거주기간은 밝히지 않았다. 이 요건을 충족한다.",
                {"parent_residence_months": 24},
                True,
            ),
        ],
    )
    group(
        "age-boundary",
        "신청자는 만 39세 이하여야 한다.",
        policy([[atom("age_years", "le", 39)]]),
        "본인의 만 나이 상한만.",
        [
            ("base", "나는 만 39세이다. 이 요건을 충족한다.", {"age_years": 39}, True),
            (
                "preserve",
                "내 만 나이는 39년이다. 이 요건을 충족한다.",
                {"age_years": 39},
                True,
            ),
            (
                "change",
                "나는 만 40세이다. 이 요건을 충족한다.",
                {"age_years": 40},
                True,
            ),
            (
                "unknown",
                "부모는 만 39세이고 내 만 나이는 밝히지 않았다. 이 요건을 충족한다.",
                {},
                True,
            ),
        ],
    )
    group(
        "and-both",
        "본인 거주기간 24개월 이상과 가구 중위소득 120% 이하를 모두 충족해야 한다.",
        policy([[atom("residence_months", "ge", 24), atom("income_pct", "le", 120)]]),
        "본인 거주기간과 가구 중위소득 비율의 연결만.",
        [
            (
                "base",
                "나는 24개월 거주했고 가구 중위소득 비율은 121%이다. 이 요건을 충족한다.",
                {"residence_months": 24, "income_pct": 121},
                True,
            ),
            (
                "preserve",
                "가구 중위소득 비율은 121%이며 나는 2년 거주했다. 이 요건을 충족한다.",
                {"residence_months": 24, "income_pct": 121},
                True,
            ),
            (
                "change",
                "나는 24개월 거주했고 가구 중위소득 비율은 120%이다. 이 요건을 충족한다.",
                {"residence_months": 24, "income_pct": 120},
                True,
            ),
            (
                "unknown",
                "나는 24개월 거주했고 가구 중위소득 비율은 밝히지 않았다. 이 요건을 충족한다.",
                {"residence_months": 24},
                True,
            ),
        ],
    )
    group(
        "or-either",
        "본인 거주기간 24개월 이상 또는 가구 중위소득 120% 이하 중 하나라도 충족하면 된다.",
        policy([[atom("residence_months", "ge", 24)], [atom("income_pct", "le", 120)]]),
        "본인 거주기간과 가구 중위소득 비율의 연결만.",
        [
            (
                "base",
                "나는 24개월 거주했고 가구 중위소득 비율은 121%이다. 이 요건을 충족한다.",
                {"residence_months": 24, "income_pct": 121},
                True,
            ),
            (
                "preserve",
                "가구 중위소득 비율은 121%이며 나는 2년 거주했다. 이 요건을 충족한다.",
                {"residence_months": 24, "income_pct": 121},
                True,
            ),
            (
                "change",
                "나는 23개월 거주했고 가구 중위소득 비율은 121%이다. 이 요건을 충족한다.",
                {"residence_months": 23, "income_pct": 121},
                True,
            ),
            (
                "unknown",
                "나는 23개월 거주했고 가구 중위소득 비율은 밝히지 않았다. 이 요건을 충족한다.",
                {"residence_months": 23},
                True,
            ),
        ],
    )
    group(
        "age-strict",
        "신청자는 만 39세 미만이어야 한다.",
        policy([[atom("age_years", "lt", 39)]]),
        "본인의 만 나이 상한만.",
        [
            ("base", "나는 만 39세이다. 이 요건을 충족한다.", {"age_years": 39}, True),
            (
                "preserve",
                "내 만 나이는 39년이다. 이 요건을 충족한다.",
                {"age_years": 39},
                True,
            ),
            (
                "change",
                "나는 만 38세이다. 이 요건을 충족한다.",
                {"age_years": 38},
                True,
            ),
            (
                "unknown",
                "부모는 만 39세이고 내 만 나이는 밝히지 않았다. 이 요건을 충족한다.",
                {},
                True,
            ),
        ],
    )
    return dict(
        bundles=bs,
        cases=cs,
        context_pairs=[
            dict(left="and-both-base", right="or-either-base", change="모두/하나라도"),
            dict(left="age-boundary-base", right="age-strict-base", change="이하/미만"),
        ],
    )


if __name__ == "__main__":
    d = build()
    (ROOT / "metamorphic.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2) + "\n"
    )
    print(len(d["bundles"]), len(d["cases"]))
