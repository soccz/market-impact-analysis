import unittest
from age_extension import extract, coverage, compile_response


class AgeTests(unittest.TestCase):
    def catalog(self, text):
        return extract(dict(scope="연령만", evidence={"E": text}))

    def test_bare_bounds(self):
        c = self.catalog("19세 이상 39세 이하")
        self.assertEqual(
            [(x["atom"]["op"], x["atom"]["value"]) for x in c["atoms"]],
            [("ge", 19), ("le", 39)],
        )
        self.assertFalse(c["warnings"])

    def test_ranges(self):
        for sep in ["~", "∼", "–", "-"]:
            c = self.catalog(f"19세 {sep} 39세")
            self.assertEqual(len(c["atoms"]), 2)

    def test_strict_range(self):
        c = self.catalog("19~39세 미만")
        self.assertIn(
            ("lt", 39), [(x["atom"]["op"], x["atom"]["value"]) for x in c["atoms"]]
        )

    def test_reversed(self):
        self.assertFalse(
            coverage(dict(scope="연령", evidence={"E": "39~19세"}))["covered"]
        )

    def test_multiple_people(self):
        self.assertFalse(
            coverage(
                dict(scope="연령", evidence={"E": "부모 65세 이상, 자녀 18세 이하"})
            )["covered"]
        )

    def test_other_scope(self):
        self.assertFalse(
            extract(dict(scope="기타", evidence={"E": "19~39세"}))["atoms"]
        )

    def test_spans(self):
        text = "신청 나이 19~39세"
        for x in self.catalog(text)["atoms"]:
            self.assertEqual(text[x["start"] : x["end"]], x["text"])


if __name__ == "__main__":
    unittest.main()
