"""Retain raw logs privately, export strict JSON, verify weights and hash public artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--backbone", type=Path, required=True)
    args = p.parse_args()
    logs = []
    weights = []
    paths = (
        list((ROOT / "results").glob("*/training.json"))
        + list((ROOT / "results_fit").glob("*/training.json"))
        + list((ROOT / "results_augmentation").glob("*/*/training.json"))
    )
    assert len(paths) == 27
    for source in sorted(paths):
        rel = source.relative_to(ROOT)
        raw = args.private / "raw_training_logs" / rel
        raw.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists():
            shutil.copy2(source, raw)
        data = json.loads(raw.read_text())
        count = sum(
            isinstance(x["grad_norm"], float) and not math.isfinite(x["grad_norm"])
            for x in data["steps"]
        )
        normalized = (
            json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode()
        assert source.read_bytes() in [
            raw.read_bytes(),
            normalized,
        ], "Unexpected log change"
        source.write_bytes(normalized)
        logs.append(
            dict(
                file=str(rel),
                raw_sha256=sha(raw),
                public_sha256=sha(source),
                nonfinite_gradient_norms=count,
            )
        )
        if rel.parts[0] == "results":
            checkpoint = args.private / "checkpoints" / source.parent.name / "model.pt"
        elif rel.parts[0] == "results_fit":
            checkpoint = (
                args.private / "fit_checkpoints" / source.parent.name / "model.pt"
            )
        else:
            checkpoint = (
                args.private
                / ("augmentation_" + rel.parts[1])
                / source.parent.name
                / "model.pt"
            )
        digest = sha(checkpoint)
        assert digest == data["checkpoint_sha256"]
        weights.append(
            dict(
                run="/".join(rel.parts[:-1]),
                sha256=digest,
                bytes=checkpoint.stat().st_size,
                public_weights=False,
            )
        )
    (ROOT / "log_serialization.json").write_text(
        json.dumps(
            dict(
                note="Raw training logs used Python JSON Infinity for overflowing gradient norms. Original bytes retained privately; public logs preserve these values as strings for strict JSON. Losses, updates, predictions and weights unchanged.",
                files=logs,
            ),
            indent=2,
        )
        + "\n"
    )
    packages = {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "transformers",
            "scikit-learn",
            "numpy",
            "scipy",
            "tokenizers",
            "matplotlib",
            "cairocffi",
            "requests",
            "beautifulsoup4",
        ]
    }
    (ROOT / "environment.json").write_text(
        json.dumps(dict(python=platform.python_version(), packages=packages), indent=2)
        + "\n"
    )
    (ROOT / "backbone_manifest.json").write_text(
        json.dumps(
            dict(
                model="klue/roberta-base",
                revision="02f94ba5e3fcb7e2a58a390b8639b0fac974a8da",
                files={
                    p.name: sha(p)
                    for p in sorted(args.backbone.iterdir())
                    if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    artifacts = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != "research_record.json"
    }

    # Reject nonstandard JSON in the final bundle.
    def invalid(value):
        raise ValueError(value)

    for p in ROOT.rglob("*.json"):
        json.loads(p.read_text(), parse_constant=invalid)
    (ROOT / "research_record.json").write_text(
        json.dumps(
            dict(
                recorded_utc=datetime.now(timezone.utc).isoformat(),
                author="soccz",
                assistance="AI-assisted design, code, synthetic data and provisional annotation; no independent human review completed",
                neural_fits=27,
                baseline_fits=1,
                checkpoint_verification=weights,
                artifacts=artifacts,
                scope="Independent personal follow-up research; no client data or delivered implementation. Synthetic data and selected official excerpts do not establish real-world generalization. Later regimes explicitly post-hoc.",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        "PASS: 27 checkpoint hashes, strict JSON logs,",
        len(artifacts),
        "public artifact hashes",
    )


if __name__ == "__main__":
    main()
