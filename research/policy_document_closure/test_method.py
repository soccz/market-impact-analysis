import copy
import unittest

from experiment import request
from method import decide, retrieve, validate


def atom(key, field, value, cmp="eq", evidence="p1"):
    return dict(
        id=key,
        op="atom",
        field=field,
        value=value,
        cmp=cmp,
        children=[],
        evidence=[evidence],
    )


def group(key, op, children, evidence="p1"):
    return dict(
        id=key,
        op=op,
        field="",
        value=0,
        cmp="eq",
        children=children,
        evidence=[evidence],
    )


def program(nodes, root="root"):
    return dict(status="ready", root=root, nodes=nodes)


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.fields = {
            "age": dict(type="number", label="나이", unit="세"),
            "worker": dict(type="boolean", label="근로자"),
            "student": dict(type="boolean", label="학생"),
        }
        self.evidence = {
            "p1": "제1조 나이 19세 이상. 근로자 또는 학생.",
            "p2": "제2조 제외의 예외는 별표 1을 따른다.",
            "p3": "별표 1 근로자인 학생은 예외다.",
        }

    def answer(self, p, facts):
        return decide(p, facts, self.fields, self.evidence)["decision"]

    def test_nested_and_or(self):
        p = program(
            [
                atom("a", "age", 19, "ge"),
                atom("w", "worker", True),
                atom("s", "student", True),
                group("or", "any", ["w", "s"]),
                group("root", "all", ["a", "or"]),
            ]
        )
        self.assertEqual(self.answer(p, dict(age=18, worker=True)), "ineligible")
        self.assertEqual(self.answer(p, dict(age=20, student=True)), "eligible")
        self.assertEqual(self.answer(p, dict(age=20, student=False)), "undetermined")

    def test_exception_to_exclusion(self):
        p = program(
            [
                atom("s", "student", True),
                atom("w", "worker", True),
                group("nw", "not", ["w"]),
                group("excluded", "all", ["s", "nw"]),
                group("root", "not", ["excluded"]),
            ]
        )
        self.assertEqual(self.answer(p, dict(student=True, worker=False)), "ineligible")
        self.assertEqual(self.answer(p, dict(student=True, worker=True)), "eligible")
        self.assertEqual(self.answer(p, dict(student=False)), "eligible")

    def test_shared_unknown_tautology(self):
        p = program(
            [
                atom("w", "worker", True),
                group("nw", "not", ["w"]),
                group("root", "any", ["w", "nw"]),
            ]
        )
        self.assertEqual(self.answer(p, {}), "eligible")

    def test_shared_unknown_contradiction(self):
        p = program(
            [
                atom("a", "age", 20, "lt"),
                atom("b", "age", 20, "ge"),
                group("root", "all", ["a", "b"]),
            ]
        )
        self.assertEqual(self.answer(p, {}), "ineligible")

    def test_missing_boundary(self):
        p = program([atom("root", "age", 20, "le")])
        for age, expected in [
            (20, "eligible"),
            (20.01, "ineligible"),
            (None, "undetermined"),
        ]:
            self.assertEqual(self.answer(p, {"age": age}), expected)

    def test_invalid_fact_type(self):
        p = program([atom("root", "age", 20, "le")])
        for value in [True, "20", -1, float("inf")]:
            self.assertEqual(self.answer(p, {"age": value}), "abstain")

    def test_cycle_rejected(self):
        p = program([group("root", "ref", ["root"])])
        self.assertEqual(self.answer(p, {}), "abstain")

    def test_unavailable_evidence_rejected(self):
        p = program([atom("root", "age", 20, "le", "fake")])
        self.assertEqual(self.answer(p, dict(age=19)), "abstain")

    def test_duplicate_and_unreachable_rejected(self):
        a = atom("root", "age", 20, "le")
        self.assertEqual(self.answer(program([a, a]), {}), "abstain")
        self.assertEqual(
            self.answer(program([a, atom("extra", "worker", True)]), {}), "abstain"
        )

    def test_reference_expansion(self):
        doc = dict(scope="제2조 제외 예외", evidence=self.evidence)
        r = retrieve(doc, k=1)
        self.assertEqual(r["ranked"], ["p2"])
        self.assertEqual(r["expanded"], ["p2", "p3"])

    def test_no_profile_or_reference_in_policy_request(self):
        doc = dict(
            scope="나이",
            fields=self.fields,
            evidence=self.evidence,
            reference_program={"SECRET": 9},
            label="SECRET",
        )
        before = request(doc, {"facts": {"age": 21}}, "qwen", "full")
        after = request(doc, {"facts": {"age": 99}}, "qwen", "full")
        self.assertEqual(before, after)
        self.assertNotIn("SECRET", str(before))
        self.assertNotIn("facts", before["messages"][1]["content"])

    def test_reference_mutation_does_not_change_request(self):
        doc = dict(scope="나이", fields=self.fields, evidence=self.evidence)
        changed = copy.deepcopy(doc)
        changed["reference_program"] = {"status": "ready"}
        for stage in ["full", "retrieved", "direct"]:
            self.assertEqual(
                request(doc, {"facts": {}}, "qwen", stage),
                request(changed, {"facts": {}}, "qwen", stage),
            )


if __name__ == "__main__":
    unittest.main()
