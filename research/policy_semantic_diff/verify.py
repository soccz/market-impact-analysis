"""Verify preserved artifacts and independently regenerate all five result files."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    record = json.loads((ROOT / "research_record.json").read_text())
    for name, digest in record["artifacts"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    inputs = json.loads((ROOT / "results/summary.json").read_text())["input_sha256"]
    for name, digest in inputs.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_engine.py"], cwd=ROOT, check=True
    )
    with tempfile.TemporaryDirectory(prefix="policy-semantic-replay-") as temp:
        subprocess.run(
            [sys.executable, "run.py", "--output", temp], cwd=ROOT, check=True
        )
        files = list((ROOT / "results").iterdir())
        for original in files:
            assert (
                original.read_bytes() == (Path(temp) / original.name).read_bytes()
            ), original.name
    print(
        json.dumps(
            {
                "artifacts": len(record["artifacts"]),
                "identical_result_files": len(files),
                "status": "PASS",
            }
        )
    )


if __name__ == "__main__":
    main()
