import unittest
from expand_retrieval import choose


class ExpansionTests(unittest.TestCase):
    def test_adjacent_enrichment_preserves_top_three_and_page_boundary(self):
        rows = [
            {
                "id": str(i),
                "page": i // 4 + 1,
                "start": i * 400,
                "end": i * 400 + 500,
                "text": ("신청 소득 기준" if i in [1, 2, 6] else "관계 없는 서류 안내")
                + str(i),
            }
            for i in range(8)
        ]
        ranked = choose("신청 소득 기준", rows, "top5")
        expanded = choose("신청 소득 기준", rows, "neighbor5")
        self.assertEqual([r["id"] for r in expanded[:3]], [r["id"] for r in ranked[:3]])
        self.assertEqual(len({r["id"] for r in expanded}), len(expanded))
        self.assertLessEqual(len(expanded), 5)
        for added in expanded[3:]:
            self.assertTrue(
                any(
                    added["page"] == seed["page"]
                    and abs(int(added["id"]) - int(seed["id"])) == 1
                    for seed in expanded[:3]
                )
            )

    def test_one_chunk_no_padding(self):
        self.assertEqual(
            len(
                choose(
                    "정책", [{"id": "1", "page": 1, "text": "정책 조건"}], "neighbor5"
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
