"""Audit real revision evidence and replay subject-specific table queries."""

import argparse
import difflib
import json
from pathlib import Path

from reader import ROOT, award_tables, load, lookup, modality, read_awards, sha, write


def key(row):
    return row["competition"], row["group"], row["rank"]


def values(rows):
    return {key(r): r["raw_amount"] for r in rows}


def change_record(a, b):
    changes = []
    for operation, i, j, k, l in difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        if operation != "equal":
            changes.append(
                dict(
                    operation=operation,
                    before_start=i,
                    before_end=j,
                    after_start=k,
                    after_end=l,
                    before_sha256=sha(a[i:j].encode()),
                    after_sha256=sha(b[k:l].encode()),
                )
            )
    return changes


def span(text, needle, document):
    # The revised full notice repeats its clause first in the change table and
    # later in the operative body. Ground the latter occurrence.
    start = text.rindex(needle)
    return dict(
        document=document,
        start=start,
        end=start + len(needle),
        sha256=sha(needle.encode()),
    )


def run(cache, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    docs = load(cache)
    sources = {s["id"]: s for s in json.loads((ROOT / "sources.json").read_text())}
    refs = json.loads((ROOT / "reference.json").read_text())
    source_ids = ["okcheon_2024_cpu", "okcheon_2024_after", "okcheon_2024_before"]
    parsed = {}
    for doc in source_ids:
        tables = award_tables(docs[doc])
        assert len(tables) == 1, (doc, len(tables))
        parsed[doc] = read_awards(tables[0], doc)
        assert len(parsed[doc]) == 18 and all(r["unit"] for r in parsed[doc])
    before, after, prior_link = [parsed[d] for d in source_ids]
    correction = "okcheon_2024_correction_notice"
    declared_tables = award_tables(docs[correction])
    assert len(declared_tables) == 2
    declared = [read_awards(t, correction) for t in declared_tables]
    assert values(before) == values(declared[0])
    assert values(after) == values(declared[1])
    assert all(
        r["amount_thousand_won"] is None for group in declared for r in group
    ), "HTML tables do not independently establish a unit"
    raw_matches = dict(
        before_cells=18,
        after_cells=18,
        declared_html_unit=None,
        unit_source="Full HWP notice near the matching table states thousand won.",
    )

    cases = []
    methods = [
        "first_numeric_cell",
        "rank_only_first_numeric_cell",
        "merged_header_table_reader",
        "current_attachments_as_history",
    ]
    for ref in refs["amounts"]:
        query = ref["competition"], ref["group"], ref["rank"]
        expected = [ref["before_thousand_won"], ref["after_thousand_won"]]
        predictions = {}
        for method in methods:
            if method == "current_attachments_as_history":
                actual = [lookup(prior_link, *query), lookup(after, *query)]
            else:
                actual = [lookup(before, *query, method), lookup(after, *query, method)]
            predictions[method] = dict(
                values=actual,
                pair_exact=actual == expected,
                relation=(
                    "unknown"
                    if None in actual
                    else ("unchanged" if actual[0] == actual[1] else "changed")
                ),
            )
        cases.append(
            dict(
                **ref,
                reference_relation=(
                    "unchanged" if expected[0] == expected[1] else "changed"
                ),
                difference_thousand_won=expected[1] - expected[0],
                methods=predictions,
                evidence={
                    "before": next(r["evidence"] for r in before if key(r) == query),
                    "after": next(r["evidence"] for r in after if key(r) == query),
                }
            )
        )

    # Read the revision's explicitly labelled old/new clause, then verify both
    # occur verbatim (after normalization) in the separate full notices.
    family_before, family_after = "okcheon_2025_before", "okcheon_2025_after"
    candidates = [
        t
        for t in docs[family_after]["tables"]
        if {"수정전", "수정후"} <= {c["text"] for c in t["cells"]}
    ]
    assert len(candidates) == 1
    table = candidates[0]
    clauses = [
        next(
            c["text"] for c in table["cells"] if c["row"] == 1 and c["col"] == col
        ).lstrip("∙")
        for col in [1, 2]
    ]
    family_evidence = [
        span(docs[doc]["text"], clause, doc)
        for doc, clause in zip([family_before, family_after], clauses)
    ]
    states = [modality(clause) for clause in clauses]
    family = dict(
        issuer_action="revision_notice",
        before=states[0],
        after=states[1],
        text_changed=clauses[0] != clauses[1],
        evidence=family_evidence,
        named_example_added=states[1]["named_example_added"]
        and not states[0]["named_example_added"],
        applicable_eligibility_change="not_established",
        reason="Both clauses retain possibility and require confirmation. Adding an example does not by itself prove a new definite exclusion.",
        reference_match=[x["state"] for x in states]
        == refs["family_modality"]["states"],
    )

    same_hash = (
        sources["okcheon_2024_before"]["sha256"]
        == sources["okcheon_2024_after"]["sha256"]
    )
    provenance = dict(
        current_prior_and_correction_hwp_identical=same_hash,
        current_attachments_distinct_historical_pair=not same_hash,
        university_old_matches_declared_old=values(before) == values(declared[0]),
        issuer_current_matches_declared_new=values(after) == values(declared[1]),
        cross_checks=raw_matches,
        note="Current byte equality does not establish when or whether a server-side replacement happened. Original posting date is not a file-version timestamp.",
    )
    diffs = {}
    for name, a, b in [
        ("talent2024", source_ids[0], source_ids[1]),
        ("family2025", family_before, family_after),
    ]:
        diffs[name] = dict(
            before=a, after=b, changes=change_record(docs[a]["text"], docs[b]["text"])
        )
    summary = dict(
        families=2,
        institutions=1,
        explicit_erratum_pairs=1,
        revision_notice_pairs=1,
        table_queries=18,
        changed=sum(c["reference_relation"] == "changed" for c in cases),
        unchanged=sum(c["reference_relation"] == "unchanged" for c in cases),
        independent_human_reviews=0,
        new_model_fits=0,
        provenance=provenance,
        family_modality=family,
        methods={
            m: dict(
                paired_values_exact=sum(c["methods"][m]["pair_exact"] for c in cases),
                relation_exact=sum(
                    c["methods"][m]["relation"] == c["reference_relation"]
                    for c in cases
                ),
                missed_changes=sum(
                    c["reference_relation"] == "changed"
                    and c["methods"][m]["relation"] == "unchanged"
                    for c in cases
                ),
                false_changes=sum(
                    c["reference_relation"] == "unchanged"
                    and c["methods"][m]["relation"] == "changed"
                    for c in cases
                ),
            )
            for m in methods
        },
        issuer_label_all_changed=dict(
            relation_exact=sum(c["reference_relation"] == "changed" for c in cases),
            false_changes=sum(c["reference_relation"] == "unchanged" for c in cases),
        ),
        interpretation="Exploratory source/table reconstruction after observing two notices, not 18 independent policy events or held-out NLP accuracy.",
    )
    for name, value in [
        ("summary.json", summary),
        ("cases.json", cases),
        ("parsed_tables.json", parsed),
        ("automatic_diffs.json", diffs),
        (
            "source_text_hashes.json",
            {k: sha(v["text"].encode()) for k, v in docs.items()},
        ),
    ]:
        write(output / name, value)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default=ROOT / "results")
    args = parser.parse_args()
    s = run(args.cache, args.output)
    print(
        json.dumps(
            {k: s[k] for k in ["families", "changed", "unchanged", "methods"]},
            ensure_ascii=False,
            indent=2,
        )
    )
