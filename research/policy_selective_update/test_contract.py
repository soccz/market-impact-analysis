import unittest
from temporal_contract import validate, predict


class ContractTests(unittest.TestCase):
    def data(self):
        return {
            "branches": [
                {
                    "from_inclusive": None,
                    "until_exclusive": "2025-03-31",
                    "max_krw": 300000,
                    "evidence": ["N"],
                },
                {
                    "from_inclusive": "2025-03-31",
                    "until_exclusive": None,
                    "max_krw": 400000,
                    "evidence": ["N"],
                },
            ]
        }

    def test_cutoff_inclusive(self):
        r = validate(self.data(), ["N"])
        self.assertEqual(predict(r, "2025-03-30", 300000), "supported")
        self.assertEqual(predict(r, "2025-03-31", 400000), "supported")

    def test_unknown_date_is_logical_not_always_unknown(self):
        r = validate(self.data(), ["N"])
        self.assertEqual(predict(r, None, 300000), "not_established")
        self.assertEqual(predict(r, None, 200000), "contradicted")

    def test_overlap_and_gap_rejected(self):
        for d in ["2025-03-30", "2025-04-01"]:
            v = self.data()
            v["branches"][1]["from_inclusive"] = d
            self.assertIsNone(validate(v, ["N"]))

    def test_bad_id_date_and_boolean_amount(self):
        self.assertIsNone(validate(self.data(), ["X"]))
        v = self.data()
        v["branches"][0]["max_krw"] = True
        self.assertIsNone(validate(v, ["N"]))
        v = self.data()
        v["branches"][1]["from_inclusive"] = "2025-02-30"
        self.assertIsNone(validate(v, ["N"]))


if __name__ == "__main__":
    unittest.main()
