import unittest
from claim_alignment import align
from source_method import coverage, compile_response


def profile(**values):
    return dict(
        status="ready",
        assertion=True,
        values=[dict(field=k, value=v) for k, v in values.items()],
    )


def vals(text, **values):
    return {v["field"]: v["value"] for v in align(text, profile(**values))[0]["values"]}


class ClaimTests(unittest.TestCase):
    def test_negative_coordination(self):
        self.assertEqual(
            vals("나는 생계·의료·주거·교육급여를 모두 받지 않는다."),
            dict(
                livelihood_benefit=False,
                medical_benefit=False,
                housing_benefit=False,
                education_benefit=False,
            ),
        )

    def test_mixed_polarity(self):
        self.assertEqual(
            vals("나는 교육급여만 받고 생계·의료·주거급여는 받지 않는다."),
            dict(
                education_benefit=True,
                livelihood_benefit=False,
                medical_benefit=False,
                housing_benefit=False,
            ),
        )

    def test_external(self):
        self.assertEqual(vals("부모는 생계·의료급여를 받지 않는다."), {})

    def test_unknown(self):
        self.assertEqual(vals("내 교육급여 수급 여부는 밝히지 않았다."), {})

    def test_conflict(self):
        self.assertEqual(
            vals("나는 교육급여를 받는다. 나는 교육급여를 받지 않는다."), {}
        )

    def test_percent(self):
        self.assertEqual(
            vals("내 가구 기준중위소득 비율은 181%이다."), dict(income_pct=181)
        )

    def test_parent_unit(self):
        self.assertEqual(
            vals("부모의 보증금은 300만원이다.", parent_residence_months=300), {}
        )
        self.assertEqual(
            vals("부모는 3개월 거주했다.", parent_residence_months=3),
            dict(parent_residence_months=3),
        )

    def test_unowned_not_patch(self):
        self.assertEqual(vals("중위소득은 180%이고 교육급여를 받는다."), {})

    def test_claim_only(self):
        first = align("나는 교육급여를 받는다.", profile())
        self.assertEqual(first, align("나는 교육급여를 받는다.", profile()))


class CoverageTests(unittest.TestCase):
    def test_nested_exception_abstains(self):
        b = dict(
            scope="age", evidence={"E": "만 39세 이하. 다만 예외 대상은 별도 판단."}
        )
        self.assertFalse(coverage(b)["covered"])
        self.assertEqual(
            compile_response(
                b, dict(status="ready", operator="ALL", conditions=["a0"])
            )["status"],
            "insufficient",
        )

    def test_unresolved_unit(self):
        self.assertFalse(
            coverage(dict(scope="cost", evidence={"E": "지원금 30만원 이하"}))[
                "covered"
            ]
        )

    def test_no_atom(self):
        self.assertFalse(
            coverage(dict(scope="city", evidence={"E": "서울 시민"}))["covered"]
        )


if __name__ == "__main__":
    unittest.main()
