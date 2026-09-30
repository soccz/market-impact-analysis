"""Check input isolation, amount identity, source offsets and representation tradeoffs."""

from copy import deepcopy
import unittest
from common import ROOT, datasets, save
from representation import money_view, project, prepare
from build_data import amounts
from model import DEFAULT_MODEL
from transformers import AutoTokenizer
import torch


class RepresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets = datasets()
        cls.tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_MODEL, local_files_only=True, use_fast=True
        )

    def test_money_values_and_alignment(self):
        examples = [
            ("1억3000만원", "130000000"),
            ("1만2000원", "12000"),
            ("5천억원", "500000000000"),
            ("50만 원", "500000"),
            ("0.5백만 원", "500000"),
            ("500,000원", "500000"),
        ]
        for text, value in examples:
            original = f"앞 {text}, 뒤."
            view = money_view(original)
            self.assertEqual(view["text"], "앞 0원, 뒤.")
            self.assertEqual(view["amounts"][0]["won"], value)
            self.assertEqual(project(view["backward"], 2, 4), (2, 2 + len(text)))

    def test_empty_passage_identity(self):
        for text in ["", "아직 확정되지 않았다.", "대상자는 2026년 12명이다."]:
            self.assertEqual(money_view(text)["text"], text)

    def test_all_source_span_roundtrips(self):
        for rows in self.sets.values():
            for row in rows:
                view = money_view(row["text"])
                for span in amounts(row["text"]):
                    a, b = project(view["forward"], span["start"], span["end"])
                    self.assertEqual(view["text"][a:b], "0원")
                    self.assertEqual(
                        project(view["backward"], a, b), (span["start"], span["end"])
                    )

    def test_identical_paired_model_inputs(self):
        audit = {}
        for a, b in [("evaluation", "units"), ("distractor", "combined")]:
            for old, new in zip(self.sets[a], self.sets[b]):
                self.assertEqual(old["pair_id"], new["pair_id"])
                self.assertEqual(
                    money_view(old["text"])["text"], money_view(new["text"])["text"]
                )
                self.assertEqual(
                    [x["won"] for x in amounts(old["text"])],
                    [x["won"] for x in amounts(new["text"])],
                )
            pa = prepare(self.sets[a], self.tokenizer, True)["encoded"][0]
            pb = prepare(self.sets[b], self.tokenizer, True)["encoded"][0]
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                self.assertTrue(torch.equal(pa[key], pb[key]))
            audit[a + ":" + b] = dict(
                pairs=len(self.sets[a]),
                identical_tokens=True,
                parsed_values_preserved=True,
            )
        save(ROOT / "input_verification.json", audit)

    def test_gold_never_changes_input(self):
        a = self.sets["evaluation"][:3]
        b = deepcopy(a)
        for row in b:
            row["relation"], row["state"] = "unspecified", "unspecified"
            row["spans"], row["evidence"] = [], []
        for normalized in [False, True]:
            pa, pb = [
                prepare(rows, self.tokenizer, normalized)["encoded"][0]
                for rows in [a, b]
            ]
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                self.assertTrue(torch.equal(pa[key], pb[key]))

    def test_hidden_numeric_distinction(self):
        a = money_view("기준 50만원, 후속 500000원")
        b = money_view("기준 50만원, 후속 600000원")
        self.assertEqual(a["text"], b["text"])
        self.assertNotEqual(
            [s["won"] for s in a["amounts"]], [s["won"] for s in b["amounts"]]
        )


if __name__ == "__main__":
    unittest.main()
