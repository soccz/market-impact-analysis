"""Independent exhaustive Boolean completion oracle and input isolation checks."""

import itertools
import unittest
from logic import interpret
from worlds import execute
from infer_explicit import request

F = ["pledge", "homeowner", "foreign_national"]


def atom(f, v):
    return dict(field=f, op="eq", value=v, evidence=["E"])


class ReasoningTests(unittest.TestCase):
    def test_all_small_boolean_dnf_completions(self):
        # 6 literals -> 21 nonempty conjunctions of length at most two.
        # Independent truth evaluation uses direct Boolean equality, not engine functions.
        literals = list(itertools.product(F, [False, True]))
        terms = [(x,) for x in literals] + list(itertools.combinations(literals, 2))
        checks = 0
        for left, right in itertools.combinations_with_replacement(terms, 2):
            terms2 = [left, right]
            dnf = [[atom(f, v) for f, v in term] for term in terms2]
            for observed in itertools.product([None, False, True], repeat=3):
                known = {f: v for f, v in zip(F, observed) if v is not None}
                unknown = [f for f in F if f not in known]
                answers = set()
                for completion in itertools.product([False, True], repeat=len(unknown)):
                    values = known | dict(zip(unknown, completion))
                    answers.add(
                        any(all(values[f] == v for f, v in term) for term in terms2)
                    )
                expected = next(iter(answers)) if len(answers) == 1 else None
                self.assertIs(execute(dnf, known)[0], expected, (terms2, known))
                checks += 1
        self.assertEqual(checks, 6237)

    def test_explicit_request_isolation(self):
        b = {
            "scope": "test",
            "evidence": {"E": "SOURCE-MARKER"},
            "reference_policy": "SECRET",
        }
        c = {"claim": "CLAIM-MARKER", "label": "SECRET", "reference_profile": "SECRET"}
        for model in ["qwen", "kanana-public"]:
            for stage in ["policy", "profile", "direct"]:
                text = str(request(b, c, model, stage)["messages"])
                self.assertNotIn("SECRET", text)
                if stage == "policy":
                    self.assertNotIn("CLAIM-MARKER", text)
                if stage == "profile":
                    self.assertNotIn("SOURCE-MARKER", text)

    def test_unknown_profile_does_not_mean_false(self):
        p = {"status": "ready", "clauses": [[atom("homeowner", False)]]}
        q = {"status": "ready", "assertion": True, "values": []}
        self.assertEqual(interpret(p, q, ["E"])["decision"], "not_established")


if __name__ == "__main__":
    unittest.main()
