"""Lock protocol, transformations, runner and upstream data before fitting."""

from datetime import datetime, timezone
from common import ROOT, V3, sha, save


def main():
    assert not (ROOT / "freeze.json").exists()
    own = [
        ROOT / x
        for x in [
            "common.py",
            "representation.py",
            "build_probe.py",
            "run.py",
            "analyze.py",
            "test_representation.py",
            "freeze.py",
            "protocol.json",
        ]
    ]
    own += list((ROOT / "data").glob("*.json*"))
    upstream = [
        V3 / x
        for x in [
            "build_data.py",
            "model.py",
            "partial_model.py",
            "evaluate.py",
            "verify.py",
            "augmentation_data/scope.jsonl",
        ]
    ]
    upstream += [
        V3 / "data" / (n + ".jsonl")
        for n in [
            "train",
            "validation",
            "evaluation",
            "units",
            "distractor",
            "official",
        ]
    ]
    import os

    save(
        ROOT / "freeze.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            note="Inputs and run settings fixed before v4 fitting; post-hoc relative to v3, not an external preregistration.",
            files={os.path.relpath(p, ROOT): sha(p) for p in own + upstream},
        ),
    )
    print("Frozen", len(own + upstream), "files")


if __name__ == "__main__":
    main()
