"""Meaningful isolation, paired-input, annotation and alignment checks before fits."""

from collections import Counter
from copy import deepcopy
import json
import unittest
from common import ROOT, datasets, save
from binding import input_view, prepare
from representation import project
from build_data import amounts
from model import DEFAULT_MODEL
from transformers import AutoTokenizer


class Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets = datasets()
        cls.tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_MODEL, local_files_only=True
        )

    def test_query_pairs_and_labels(self):
        for name, rows in self.sets.items():
            if name == "absent":
                continue
            for a, b in zip(rows[::2], rows[1::2]):
                self.assertEqual(a["text"], b["text"])
                self.assertNotEqual(a["query"], b["query"])
                self.assertNotEqual(
                    (a["relation"], a["state"]), (b["relation"], b["state"])
                )
                self.assertEqual(
                    [s["role"] == "other" for s in a["spans"]],
                    [s["role"] != "other" for s in b["spans"]],
                )
            self.assertEqual(
                len(
                    set(
                        Counter(r["relation"] + "/" + r["state"] for r in rows).values()
                    )
                ),
                1,
            )
            self.assertEqual(Counter(r["position"] for r in rows)[0], len(rows) // 2)

    def test_offsets_and_values(self):
        for rows in self.sets.values():
            for r in rows:
                found = amounts(r["text"])
                self.assertEqual(len(found), 4)
                for s, t in zip(r["spans"], found):
                    self.assertEqual({k: s[k] for k in t}, t)
                for e in r["evidence"]:
                    self.assertEqual(r["text"][e["start"] : e["end"]], e["text"])
                for mode in ["plain", "marked", "blind"]:
                    v = input_view(r["query"], r["text"], mode)
                    self.assertEqual(len(v["forward"]), len(r["text"]))
                    self.assertEqual(len(v["backward"]), len(v["text"]))
                    for s in r["spans"] + r["evidence"]:
                        a, b = project(v["forward"], s["start"], s["end"])
                        self.assertEqual(
                            project(v["backward"], a, b), (s["start"], s["end"])
                        )

    def test_no_gold_enters_inputs(self):
        rows = deepcopy(self.sets["evaluation"][:2])
        changed = deepcopy(rows)
        for r in changed:
            r.update(
                relation="unspecified",
                state="unspecified",
                entities=["invented"],
                evidence=[],
            )
            for sp in r["spans"]:
                sp["role"] = "other"
        for mode in ["plain", "marked", "blind"]:
            a, b = [
                prepare(x, self.tokenizer, mode)["encoded"][0] for x in [rows, changed]
            ]
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                self.assertTrue(a[key].equal(b[key]))

    def test_blind_identity_and_unit_invariance(self):
        for mode in ["plain", "marked", "blind"]:
            base = [
                input_view(r["query"], r["text"], mode) for r in self.sets["evaluation"]
            ]
            units = [
                input_view(r["query"], r["text"], mode) for r in self.sets["units"]
            ]
            self.assertEqual(
                [(v["query"], v["text"]) for v in base],
                [(v["query"], v["text"]) for v in units],
            )
            for a, b in zip(base[::2], base[1::2]):
                if mode == "blind":
                    self.assertEqual((a["query"], a["text"]), (b["query"], b["text"]))
                else:
                    self.assertNotEqual(
                        (a["query"], a["text"]), (b["query"], b["text"])
                    )

    def test_document_split_and_actual_length(self):
        seen = set()
        audit = {}
        for name, rows in self.sets.items():
            passages = {r["text"] for r in rows}
            if name in ["train", "validation", "evaluation"]:
                self.assertFalse(passages & seen)
                seen |= passages
            audit[name] = {}
            for mode in ["plain", "marked", "blind"]:
                x = prepare(rows, self.tokenizer, mode)
                distinct = set()
                for r, v in zip(rows, x["views"]):
                    text = v["text"]
                    query = v["query"]
                    for i, entity in enumerate(r["entities"]):
                        text = text.replace(entity, f"사업{i}")
                        query = query.replace(entity, f"사업{i}")
                    distinct.add((query, text))
                audit[name][mode] = dict(
                    max_tokens=int(x["encoded"][0]["attention_mask"].sum(1).max()),
                    normalized_expressions=len(distinct),
                )
        save(ROOT / "input_audit.json", audit)


if __name__ == "__main__":
    unittest.main()
