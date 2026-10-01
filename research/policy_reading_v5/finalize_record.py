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
    weights = []
    for condition in ["plain", "marked", "blind"]:
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
            new_neural_fits=9,
            reused_v4_models=3,
            reused_weight_input_views=2,
            post_result_routing_fits=0,
            post_result_routing_uses_same_three_v4_weights=True,
            v3_v4_v5_retained_fits=42,
            checkpoints=weights,
            artifacts={str(p.relative_to(ROOT)): sha(p) for p in files},
            limits="Post-hoc follow-up to v4 failures; new authored paired-query grammar and layout rewrites. No new official cases. No client material, new human gold, model novelty or real-world generalization claim.",
        ),
    )
    print(
        "PASS nine actual checkpoint hashes and", len(files), "public artifact hashes"
    )


if __name__ == "__main__":
    main()
