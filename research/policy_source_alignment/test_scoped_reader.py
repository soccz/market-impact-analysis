import unittest
from bridge import sha
from scoped_reader import read
from age_extension import extract


class ReaderTests(unittest.TestCase):
    def call(self, text, age=20, mode="bounded_clause", digest=None, fields=None):
        b = dict(scope="연령만", evidence={"E": text})
        ids = [a["id"] for a in extract(b)["atoms"]]
        contract = dict(
            mode=mode,
            evidence_sha256=digest or sha(b["evidence"]),
            fields=["age_years"] if fields is None else fields,
        )
        p = dict(
            status="ready",
            assertion=True,
            values=[] if age is None else [dict(field="age_years", value=age)],
        )
        return read(
            b,
            dict(status="ready", operator="ALL", conditions=ids),
            "내 나이 정보.",
            p,
            contract,
        )

    def test_independent_interval_grid(self):
        # Independent scalar truth table; includes unknown rather than treating it false.
        for lo, hi in [(18, 60), (19, 39), (60, 65), (65, 100)]:
            for age in [None, *range(0, 111)]:
                expected = (
                    "not_established"
                    if age is None
                    else "supported" if lo <= age <= hi else "contradicted"
                )
                self.assertEqual(
                    self.call(f"{lo}세 이상 {hi}세 이하", age)["decision"], expected
                )

    def test_whole_program_refused(self):
        self.assertEqual(
            self.call("19세 이상 39세 이하", mode="full_program")["reason"],
            "explicit_bounded_scope_required",
        )

    def test_changed_source_refused(self):
        self.assertEqual(
            self.call("19세 이상 39세 이하", digest="stale")["reason"],
            "source_version_changed",
        )

    def test_unresolved_negation(self):
        self.assertEqual(
            self.call("19세 이상이라고 해서 충족하는 것은 아니다")["reason"],
            "unresolved_negation_or_override",
        )

    def test_nested_coordination(self):
        self.assertEqual(
            self.call("19세 이상 및 39세 이하 또는 만 65세 이상")["reason"],
            "nested_coordination_outside_one_operator",
        )

    def test_wrong_field(self):
        self.assertEqual(
            self.call("19세 이상", fields=["deposit_krw"])["reason"],
            "field_outside_declared_scope",
        )


if __name__ == "__main__":
    unittest.main()
