import unittest

from routing import audit_selected, updated


class AuditTests(unittest.TestCase):
    def test_audit_selection_is_fixed_without_labels(self):
        ids = [f"case-{i}" for i in range(50)]
        first = {x: audit_selected(x) for x in ids}
        self.assertEqual(first, {x: audit_selected(x) for x in reversed(ids)})
        self.assertGreater(sum(first.values()), 0)
        self.assertLess(sum(first.values()), len(ids))

    def test_unknown_cache_is_refreshed_by_audited_map(self):
        b = {"before": {"O1": "old"}, "after": {"N1": "new"}}
        old = {"decision": "not_established", "evidence": ["O1"]}
        fresh = {"decision": "supported", "evidence": ["N1"]}
        r = updated(
            "audited_change_map",
            old,
            fresh,
            b,
            {"route": "keep", "evidence": ["N1"]},
            "case",
        )
        self.assertTrue(r["refreshed"])
        self.assertEqual(r["prediction"], fresh)


if __name__ == "__main__":
    unittest.main()
