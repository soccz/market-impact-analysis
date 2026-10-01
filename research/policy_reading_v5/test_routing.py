"""Test source-only routing, no cross-gap evidence inflation and unsupported cases."""

from copy import deepcopy
import unittest
from common import datasets
from routing import retrieve, source_parts, prepare
from model import DEFAULT_MODEL
from transformers import AutoTokenizer


class Routing(unittest.TestCase):
    def test_sentence_selection(self):
        text = "가람은 50만 원이다. 별빛은 80만 원이다. 가람의 정정은 확정됐다."
        result = retrieve("가람", text)
        self.assertEqual(result["text"], "가람은 50만 원이다. 가람의 정정은 확정됐다.")
        self.assertTrue(result["answerable"])
        self.assertNotIn("별빛", result["text"])
        ranges = source_parts(result["back"], 0, len(result["back"]))
        self.assertEqual(len(ranges), 2)
        self.assertNotIn("별빛", "".join(text[a:b] for a, b in ranges))

    def test_missing_reference_and_known_substring_limit(self):
        self.assertFalse(retrieve("가람", "별빛은 80만 원이다.")["answerable"])
        self.assertFalse(
            retrieve(
                "가람", "가람 및 별빛을 설명한다. 앞서 언급한 사업은 50만 원이다."
            )["answerable"]
        )
        # This deliberately records an unresolved exact-substring limitation.
        self.assertTrue(retrieve("가람", "새가람은 50만 원이다.")["answerable"])

    def test_decimal_boundary(self):
        v = retrieve("가람", "가람은 1.5만 원이다. 별빛은 80만 원이다.")
        self.assertEqual(v["text"], "가람은 1.5만 원이다.")

    def test_model_inputs_ignore_gold_and_distant_evidence(self):
        rows = datasets()["interleaved"][:2]
        changed = deepcopy(rows)
        for r in changed:
            r.update(
                evidence=[],
                spans=[],
                relation="unspecified",
                state="denied",
                entities=[],
                target=None,
            )
        tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True)
        a, b = [prepare(rr, tok) for rr in [rows, changed]]
        self.assertEqual(a["rows"], b["rows"])
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            self.assertTrue(
                a["encoded"]["encoded"][0][key].equal(b["encoded"]["encoded"][0][key])
            )


if __name__ == "__main__":
    unittest.main()
