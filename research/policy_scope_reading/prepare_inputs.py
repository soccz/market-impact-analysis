"""Rebuild private inputs without rewriting frozen public annotation records."""

import argparse
import json
from pathlib import Path

from infer import ROOT, digest
from build_official import build
from build_replication import build as replication_build
from build_followup import build as followup_build


def prepare(cache):
    cache = Path(cache)
    rows = build(cache)
    (cache / "official_inputs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    assert (
        digest((cache / "official_inputs.json").read_bytes())
        == json.loads((ROOT / "test_freeze.json").read_text())[
            "official_private_input_sha256"
        ]
    )
    replication_build(cache)
    followup_build(cache)
    for split, freeze in [
        ("replication", "replication_freeze.json"),
        ("followup", "followup_freeze.json"),
    ]:
        assert (
            digest((cache / (split + "_inputs.json")).read_bytes())
            == json.loads((ROOT / freeze).read_text())["private_input_sha256"]
        )
    print(dict(official=54, replication=36, followup=12, frozen_inputs_match=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    prepare(p.parse_args().cache)
