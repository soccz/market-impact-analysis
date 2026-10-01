"""Meaningful failure boundaries of the exploratory table/modality readers."""

import unittest

from reader import cell, expand, lookup, modality, read_awards


def sample():
    cells = [
        cell("구분", 0, 0, rowspan=3),
        cell("전국대회", 0, 1, colspan=3),
        cell("국제대회", 0, 4, colspan=3),
    ]
    for start in [1, 4]:
        cells += [
            cell("개인", 1, start, rowspan=2),
            cell("단체", 1, start + 1, colspan=2),
            cell("2~4명", 2, start + 1),
            cell("5명 이상", 2, start + 2),
        ]
    for rank in [1, 2, 3]:
        cells.append(cell(str(rank) + "위", rank + 2, 0))
        for col in range(1, 7):
            cells.append(cell(str(rank * 100 + col), rank + 2, col))
    return dict(index=0, cells=cells, unit=dict(multiplier_won=1000))


class ReaderTests(unittest.TestCase):
    def test_merged_headers_link_the_right_recipient(self):
        rows = read_awards(sample(), "synthetic")
        self.assertEqual(len(rows), 18)
        self.assertEqual(lookup(rows, "국제대회", "단체2~4명", 2), 205)

    def test_same_numeric_cells_different_headers_change_answers(self):
        table = sample()
        before = read_awards(table, "synthetic")
        table["cells"][1], table["cells"][2] = cell("국제대회", 0, 1, colspan=3), cell(
            "전국대회", 0, 4, colspan=3
        )
        after = read_awards(table, "synthetic")
        self.assertEqual(
            sorted(r["raw_amount"] for r in before),
            sorted(r["raw_amount"] for r in after),
        )
        self.assertEqual(lookup(before, "전국대회", "개인", 1), 101)
        self.assertEqual(lookup(after, "전국대회", "개인", 1), 104)

    def test_missing_recipient_header_never_fills_by_position(self):
        table = sample()
        table["cells"] = [
            c for c in table["cells"] if not (c["row"] == 1 and c["col"] == 5)
        ]
        rows = read_awards(table, "synthetic")
        self.assertIsNone(lookup(rows, "국제대회", "단체2~4명", 2))
        self.assertEqual(lookup(rows, "국제대회", "개인", 2), 204)

    def test_unknown_unit_prevents_money_answer(self):
        table = sample()
        table["unit"] = None
        rows = read_awards(table, "synthetic")
        self.assertEqual(rows[0]["raw_amount"], 101)
        self.assertIsNone(lookup(rows, "전국대회", "개인", 1))

    def test_incomplete_query_abstains(self):
        rows = read_awards(sample(), "synthetic")
        self.assertIsNone(lookup(rows, "전국대회", None, 1))
        self.assertIsNone(lookup(rows, "전국대회", "개인", 4))

    def test_overlapping_cells_fail_closed(self):
        with self.assertRaises(ValueError):
            expand([cell("a", 0, 0, colspan=2), cell("b", 0, 1)])

    def test_possibility_is_not_definite_exclusion(self):
        clause = "다른 장학금 신청자는 제외 대상이 될 수 있으므로 신청 전 확인"
        for suffix in ["", " (댐주변지역 학생 장학금 등)"]:
            state = modality(clause + suffix)
            self.assertEqual(state["state"], "needs_confirmation")
            self.assertFalse(state["definite_exclusion"])
        self.assertEqual(
            modality("지원 대상에서 제외합니다")["state"], "excluded_by_this_clause"
        )
        self.assertEqual(
            modality("제외 대상이 아닙니다")["state"], "not_excluded_by_this_clause"
        )
        self.assertEqual(modality("지원 여부는 별첨 자료 참조")["state"], "unknown")


if __name__ == "__main__":
    unittest.main()
