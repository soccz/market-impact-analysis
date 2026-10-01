"""Freeze the design and executable inputs before any v5 fit or inference."""

from datetime import datetime, timezone
import os
from common import ROOT, V3, V4, save, sha


def main():
    assert not (ROOT / "freeze.json").exists()
    paths = list(ROOT.glob("*.py")) + [
        ROOT / "protocol.json",
        ROOT / "input_audit.json",
    ]
    paths += list((ROOT / "data").glob("*"))
    paths += [
        V3 / n
        for n in [
            "model.py",
            "partial_model.py",
            "build_data.py",
            "evaluate.py",
            "verify.py",
        ]
    ]
    paths += [V4 / "representation.py"]
    paths += [
        V4 / "results/normalized_scope" / str(s) / "training.json"
        for s in [17, 42, 2026]
    ]
    save(
        ROOT / "freeze.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            note="Local pre-run lock after v4 observations; not external preregistration.",
            files={os.path.relpath(p, ROOT): sha(p) for p in paths},
        ),
    )
    print("Frozen", len(paths), "inputs")


if __name__ == "__main__":
    main()
