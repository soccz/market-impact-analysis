"""Public, synthetic explanations for the September 2026 research note.

All documents, times, dependencies and companies are invented and supplied by
hand. This is not a news extractor, the private ledger, or a market model.
Run with --check to verify replay boundaries and dependency-local changes.
"""

import argparse
import json


DOCUMENTS = [
    {
        "id": "notice",
        "title": "공식 발표: 사업군 A·B 포함",
        "published": "09:00",
        "seen": "09:00",
        "origin": "notice",
        "scope": ["A", "B"],
        "kind": "발표",
    },
    {
        "id": "copy_a",
        "title": "발표문을 전재한 기사 1",
        "published": "09:06",
        "seen": "09:10",
        "origin": "notice",
        "kind": "전재",
    },
    {
        "id": "copy_b",
        "title": "같은 발표를 전한 기사 2",
        "published": "09:08",
        "seen": "09:10",
        "origin": "notice",
        "kind": "전재",
    },
    {
        "id": "correction",
        "title": "명시적 정정: 사업군 B 제외",
        "published": "09:25",
        "seen": "09:30",
        "origin": "notice",
        "scope": ["A"],
        "kind": "정정",
    },
    {
        "id": "late_guide",
        "title": "뒤늦게 확인한 별도 시행 안내",
        "published": "09:05",
        "seen": "09:45",
        "origin": "guide",
        "kind": "늦은 확인",
    },
]
CUTOFFS = ["09:00", "09:10", "09:30", "09:45"]
HYPOTHESES = [
    {
        "id": "A",
        "company": "가상 기업 A",
        "path": "사업군 A → 지원 자격",
        "fields": ["eligibility_A"],
        "origins": ["notice"],
    },
    {
        "id": "B",
        "company": "가상 기업 B",
        "path": "사업군 B → 지원 자격",
        "fields": ["eligibility_B"],
        "origins": ["notice"],
    },
    {
        "id": "C",
        "company": "가상 기업 C",
        "path": "별도 정책 → 원료 비용",
        "fields": ["other_policy_cost"],
        "origins": ["other_policy"],
    },
]


def minute(value):
    hour, minutes = map(int, value.split(":"))
    if not (0 <= hour < 24 and 0 <= minutes < 60):
        raise ValueError("same-day HH:MM required")
    return hour * 60 + minutes


def replay(documents, cutoff):
    """Replay what this fictional collector had observed, not market knowledge."""
    if len({doc["id"] for doc in documents}) != len(documents):
        raise ValueError("unique document IDs required")
    if any(minute(doc["seen"]) < minute(doc["published"]) for doc in documents):
        raise ValueError("observation cannot precede publication")
    visible = sorted(
        (doc for doc in documents if minute(doc["seen"]) <= minute(cutoff)),
        key=lambda doc: minute(doc["seen"]),
    )
    # Only the hand-authored notice/correction rows carry a scope update.
    updates = [doc for doc in visible if "scope" in doc]
    return {
        "cutoff": cutoff,
        "documents": visible,
        "document_count": len(visible),
        "origin_count": len({doc["origin"] for doc in visible}),
        "scope": updates[-1]["scope"] if updates else None,
        "scope_revisions": max(0, len(updates) - 1),
    }


def dependencies(changed_fields=(), unavailable_origins=()):
    """Flag only the supplied dependencies; do not infer economic effects."""
    rows = []
    for hypothesis in HYPOTHESES:
        missing = sorted(set(hypothesis["origins"]) & set(unavailable_origins))
        changed = sorted(set(hypothesis["fields"]) & set(changed_fields))
        status = "hold" if missing else "review" if changed else "retain"
        rows.append(
            {**hypothesis, "status": status, "changed": changed, "missing": missing}
        )
    return rows


def build_report():
    return {
        "mode": "new_synthetic_research_explainer",
        "manual_inputs": True,
        "replay": [replay(DOCUMENTS, cutoff) for cutoff in CUTOFFS],
        "dependencies": {
            "baseline": dependencies(),
            "correction": dependencies(changed_fields=["eligibility_B"]),
            "source_missing": dependencies(unavailable_origins=["notice"]),
        },
    }


def verify():
    report = build_report()
    states = report["replay"]
    assert [s["document_count"] for s in states] == [1, 3, 4, 5]
    assert [s["origin_count"] for s in states] == [1, 1, 1, 2]
    assert [s["scope_revisions"] for s in states] == [0, 0, 1, 1]
    assert [s["scope"] for s in states] == [["A", "B"], ["A", "B"], ["A"], ["A"]]
    assert replay(DOCUMENTS, "08:59")["scope"] is None
    assert replay(DOCUMENTS, "09:29")["scope"] == ["A", "B"]
    assert "late_guide" not in [
        doc["id"] for doc in replay(DOCUMENTS, "09:44")["documents"]
    ]
    # Appending later information must not rewrite an earlier snapshot.
    for end in range(1, len(DOCUMENTS) + 1):
        earlier = DOCUMENTS[end - 1]["seen"]
        # Treat equal observation times as a complete batch.
        prefix = [doc for doc in DOCUMENTS if minute(doc["seen"]) <= minute(earlier)]
        assert replay(prefix, earlier) == replay(DOCUMENTS, earlier)
    assert [row["status"] for row in report["dependencies"]["correction"]] == [
        "retain",
        "review",
        "retain",
    ]
    assert [row["status"] for row in report["dependencies"]["source_missing"]] == [
        "hold",
        "hold",
        "retain",
    ]
    assert all(
        row["status"] == "retain"
        for row in dependencies(changed_fields=["unrelated_field"])
    )
    both = dependencies(
        changed_fields=["eligibility_B"], unavailable_origins=["notice"]
    )
    assert both[1]["status"] == "hold"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = verify()
    if args.check:
        print(
            "PASS: synthetic replay, late arrival, historical stability and selective dependency review"
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
