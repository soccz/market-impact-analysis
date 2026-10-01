"""Post-freeze arithmetic verification, not additional language-model examples."""

from decimal import Decimal
import unittest
from grounding import money


class NotationGridTests(unittest.TestCase):
    def test_value_preserved_across_630_canonical_formats(self):
        checked = 0
        for hundred_millions in [0, 1, 2, 9, 100]:
            for ten_thousands in [0, 1, 24, 1250, 2400, 9999]:
                for remainder in [0, 1, 9, 99, 240, 2400, 9999]:
                    value = (
                        hundred_millions * 100000000 + ten_thousands * 10000 + remainder
                    )
                    for text in [
                        f"{value:,}원",
                        f"{value // 10000}만 {value % 10000}원",
                        f"{Decimal(value) / Decimal(100000000):f}억원",
                    ]:
                        self.assertEqual(money(text), value, text)
                        checked += 1
        self.assertEqual(checked, 630)


if __name__ == "__main__":
    unittest.main()
