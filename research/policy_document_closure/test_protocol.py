import json
from pathlib import Path
import unittest

from prepare_review import packet

ROOT = Path(__file__).parent


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "reference.json").read_text())

    def test_meaning_pairs_keep_facts_and_other_paragraphs(self):
        for pair, paragraph in [("exception", "p04"), ("subject", "p02")]:
            a, b = [d for d in self.data["documents"] if d.get("pair") == pair]
            self.assertEqual(a["fields"], b["fields"])
            changed = [
                key for key in a["evidence"] if a["evidence"][key] != b["evidence"][key]
            ]
            self.assertEqual(changed, [paragraph])
            ca = [c for c in self.data["cases"] if c["document"] == a["id"]]
            cb = [c for c in self.data["cases"] if c["document"] == b["id"]]
            self.assertEqual([c["facts"] for c in ca], [c["facts"] for c in cb])
            self.assertEqual(sum(x["label"] != y["label"] for x, y in zip(ca, cb)), 3)

    def test_blind_packets_exclude_labels_and_model_results(self):
        a = packet(self.data, 20261002)
        b = packet(self.data, 20261003)
        self.assertEqual({r["case_id"] for r in a}, {r["case_id"] for r in b})
        self.assertNotEqual([r["case_id"] for r in a], [r["case_id"] for r in b])
        for row in a + b:
            self.assertEqual(
                set(row),
                {
                    "review_id",
                    "case_id",
                    "document",
                    "scope",
                    "fields",
                    "facts",
                    "evidence",
                },
            )
            self.assertNotIn("reference_program", str(row))
            self.assertNotIn("independent_label", str(row))


if __name__ == "__main__":
    unittest.main()
