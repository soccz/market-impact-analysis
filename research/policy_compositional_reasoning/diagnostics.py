"""Descriptive ablations, abstention tradeoffs and explanations; no new model calls."""

import json
from collections import Counter
from pathlib import Path
from explain import explain
from infer import sha, MODELS

ROOT = Path(__file__).parent
STANDARD = [
    "development",
    "development_explicit",
    "development_examples",
    "official",
    "official_examples",
    "composition",
    "composition_examples",
]
STANDARD += ["official_repair", "composition_repair"]
SPLITS = STANDARD + ["capability_base", "capability_extended", "capability_repair"]


def read(n):
    return json.loads((ROOT / n).read_text())


def dataset(split):
    return read(
        "development.json"
        if split.startswith("development")
        else (
            "official_reference.json"
            if split.startswith("official")
            else (
                "composition.json"
                if split.startswith("composition")
                else "capability_reference.json"
            )
        )
    )


def calculate():
    out = {}
    explanations = []
    unique = {}
    total = 0
    for split in SPLITS:
        rows = read(f"results/{split}/rows.json")
        preds = read(f"results/{split}/predictions.json")
        data = dataset(split)
        bs = {b["id"]: b for b in data["bundles"]}
        cs = {c["id"]: c for c in data["cases"]}
        lookup = {(r["model"], r["stage"], r["id"]): r for r in preds}
        for p in preds:
            k = (
                p["model"],
                p["stage"],
                p["id"],
                p["request_sha256"],
                p["created_utc"],
                p["response_sha256"],
            )
            unique[k] = p
            total += 1
        out[split] = {}
        for model in MODELS:
            rr = [r for r in rows if r["model"] == model]

            def selective(accepted):
                good = sum(r["pipeline"]["decision"] == r["label"] for r in accepted)
                return dict(
                    covered=len(accepted),
                    n=len(rr),
                    correct=good,
                    wrong=len(accepted) - good,
                    accuracy=good / len(accepted) if accepted else None,
                    review=len(rr) - len(accepted),
                    wrong_ids=[
                        r["id"]
                        for r in accepted
                        if r["pipeline"]["decision"] != r["label"]
                    ],
                )

            accepted = [
                r
                for r in rr
                if r["pipeline"]["decision"] == r["direct"]["decision"]
                and r["pipeline"]["decision"] in ["supported", "contradicted"]
            ]
            full = [
                r
                for r in rr
                if r["pipeline"]["decision"] in ["supported", "contradicted"]
            ]
            failures = [r for r in rr if r["pipeline"]["decision"] != r["label"]]
            out[split][model] = dict(
                agreement_gate=selective(accepted),
                all_determinate=selective(full),
                policy_replacement_fixes=sum(
                    r["reference_policy"]["decision"] == r["label"] for r in failures
                ),
                profile_replacement_fixes=sum(
                    r["reference_profile"]["decision"] == r["label"] for r in failures
                ),
                failures=len(failures),
                by_bundle={
                    b["id"]: dict(
                        n=sum(r["bundle"] == b["id"] for r in rr),
                        direct=sum(
                            r["bundle"] == b["id"]
                            and r["direct"]["decision"] == r["label"]
                            for r in rr
                        ),
                        pipeline=sum(
                            r["bundle"] == b["id"]
                            and r["pipeline"]["decision"] == r["label"]
                            for r in rr
                        ),
                    )
                    for b in data["bundles"]
                },
                comparison_calls=dict(
                    direct=len(data["cases"]),
                    pipeline=len(data["bundles"]) + len(data["cases"]),
                    call_reuse_assumption="one policy extraction per bundle, one profile parse per claim; execution adds no LLM call",
                ),
            )
            if split in STANDARD:
                changes = [
                    r
                    for r in rr
                    if r["pipeline"]["decision"] != r["kleene"]["decision"]
                ]
                out[split][model]["finite_partition"] = dict(
                    changed=len(changes),
                    fixed=sum(
                        r["pipeline"]["decision"] == r["label"]
                        and r["kleene"]["decision"] != r["label"]
                        for r in changes
                    ),
                    harmed=sum(
                        r["pipeline"]["decision"] != r["label"]
                        and r["kleene"]["decision"] == r["label"]
                        for r in changes
                    ),
                    both_wrong=sum(
                        r["pipeline"]["decision"] != r["label"]
                        and r["kleene"]["decision"] != r["label"]
                        for r in changes
                    ),
                    cases=[
                        dict(
                            id=r["id"],
                            label=r["label"],
                            kleene=r["kleene"]["decision"],
                            finite=r["pipeline"]["decision"],
                        )
                        for r in changes
                    ],
                )
                for r in rr:
                    b = bs[r["bundle"]]
                    p = lookup[model, "policy", r["bundle"]]["parsed"]
                    q = lookup[model, "profile", r["id"]]["parsed"]
                    explanations.append(
                        dict(
                            split=split,
                            model=model,
                            id=r["id"],
                            predicted=explain(p, q, b["evidence"]),
                            reference=explain(
                                b["reference_policy"],
                                cs[r["id"]]["reference_profile"],
                                b["evidence"],
                            ),
                        )
                    )
    records = list(unique.values())
    counts = Counter((p["model"], p["stage"]) for p in records)
    accounting = dict(
        unique_saved_responses=len(records),
        exported_entries_including_reuse=total,
        reused_entries=total - len(records),
        by_model_stage={f"{m}/{s}": n for (m, s), n in sorted(counts.items())},
        authored_development_claims=24,
        authored_composition_claims=32,
        new_document_primary_claims=32,
        same_document_capability_claims=16,
        total_base_claims=104,
        document_families=2,
        new_document_files=2,
        independent_human_reviewers=0,
        new_training_runs=0,
        interruption_note="개발 최초 runner가 SIGTERM으로 중단되어 저장된 응답 이후 재개했다. 보존 응답 수이며 전송 후 저장 전에 중단된 미보존 요청의 개수는 알 수 없다. 복사 재사용은 추가 호출로 세지 않는다.",
    )
    return out, explanations, accounting


if __name__ == "__main__":
    out, explanations, accounting = calculate()
    for name, data in [
        ("diagnostics", out),
        ("explanations", explanations),
        ("accounting", accounting),
    ]:
        (ROOT / "results" / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(accounting, ensure_ascii=False))
