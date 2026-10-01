import unittest
from provenance import covered
from prepare_inputs import public_stub
from core import changed_old_ids


class ProvenanceTests(unittest.TestCase):
    def test_overlapping_chunks_cover_without_double_count(self):
        loc = {
            "a": {"document": "d", "page": 1, "start": 0, "end": 500},
            "b": {"document": "d", "page": 1, "start": 400, "end": 900},
        }
        target = [{"document": "d", "page": 1, "start": 450, "end": 800}]
        self.assertTrue(covered(loc, ["b"], target))
        self.assertTrue(covered(loc, ["a", "b"], target))
        self.assertFalse(covered(loc, ["a"], target))

    def test_other_document_cannot_cover(self):
        self.assertFalse(
            covered(
                {"a": {"document": "old", "page": 1, "start": 0, "end": 900}},
                ["a"],
                [{"document": "new", "page": 1, "start": 1, "end": 4}],
            )
        )

    def test_hash_stub_preserves_equality(self):
        b = {
            "id": "x",
            "source_locators": {
                "before": {"O1": {"text_sha256": "a"}, "O2": {"text_sha256": "b"}},
                "after": {"N1": {"text_sha256": "a"}},
            },
        }
        self.assertEqual(
            changed_old_ids(public_stub({"bundles": [b], "cases": []})["bundles"][0]),
            {"O2"},
        )


if __name__ == "__main__":
    unittest.main()
