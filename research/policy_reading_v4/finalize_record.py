"""Hash actual retained weights and all public artifacts without exporting weights."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from common import ROOT, sha, save, check_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    args = parser.parse_args()
    from pathlib import Path

    private = Path(args.checkpoints)
    check_lock()
    from common import datasets
    from representation import money_view

    audit = {}
    for scope in [False, True]:
        data = datasets(scope)
        audit["scope" if scope else "base"] = {
            name: dict(
                rows=len(rows),
                normalized_query_masked_strings=len(
                    {
                        money_view(r["text"])["text"].replace(r["query"], "TARGET")
                        for r in rows
                    }
                ),
            )
            for name, rows in data.items()
        }
    save(ROOT / "data_audit.json", audit)
    weights = []
    for condition in ["normalized", "normalized_scope"]:
        for seed in [17, 42, 2026]:
            log = json.loads(
                (ROOT / f"results/{condition}/{seed}/training.json").read_text()
            )
            p = private / condition / str(seed) / "model.pt"
            digest = sha(p)
            assert digest == log["checkpoint_sha256"]
            weights.append(
                dict(
                    condition=condition,
                    seed=seed,
                    sha256=digest,
                    bytes=p.stat().st_size,
                    public=False,
                )
            )
    save(
        ROOT / "environment.json",
        dict(
            python=platform.python_version(),
            packages={
                n: importlib.metadata.version(n)
                for n in [
                    "torch",
                    "transformers",
                    "numpy",
                    "scikit-learn",
                    "matplotlib",
                    "cairocffi",
                ]
            },
        ),
    )
    files = [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != "research_record.json"
    ]
    assert not any(p.suffix in [".pt", ".bin", ".pyc", ".pkl"] for p in files)

    def invalid(value):
        raise ValueError(value)

    for p in files:
        if p.suffix == ".json":
            json.loads(p.read_text(), parse_constant=invalid)
    save(
        ROOT / "research_record.json",
        dict(
            recorded_utc=datetime.now(timezone.utc).isoformat(),
            author="soccz",
            assistance="AI-assisted study design, implementation, synthetic probes, provisional annotation and prose; human reviews completed: 0.",
            new_neural_fits=6,
            reused_v3_models=6,
            checkpoints=weights,
            artifacts={str(p.relative_to(ROOT)): sha(p) for p in files},
            limits="Post-hoc follow-up to observed v3 failures; same authored grammar and selected official provisional labels. No client material, new human gold, model novelty or real-world generalization claim.",
        ),
    )
    print("PASS six actual checkpoint hashes and", len(files), "public artifact hashes")


if __name__ == "__main__":
    main()
