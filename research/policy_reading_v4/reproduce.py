"""Re-fit only the six new conditions, retaining published raw comparators."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from common import ROOT, V3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    assert not work.exists(), "Choose a fresh directory"
    assert args.model.is_dir()
    model = args.model.resolve()
    work.mkdir(parents=True)
    shutil.copytree(V3, work / V3.name, ignore=shutil.ignore_patterns("__pycache__"))
    target = work / ROOT.name
    target.mkdir()
    for source in ROOT.glob("*.py"):
        shutil.copy2(source, target / source.name)
    for name in ["protocol.json", "freeze.json"]:
        shutil.copy2(ROOT / name, target / name)
    shutil.copytree(ROOT / "data", target / "data")
    for condition in ["raw", "raw_scope"]:
        shutil.copytree(ROOT / "results" / condition, target / "results" / condition)
    if args.prepare_only:
        print("Prepared", target)
        return
    for condition in ["normalized", "normalized_scope"]:
        subprocess.run(
            [
                sys.executable,
                "run.py",
                "--condition",
                condition,
                "--model",
                str(model),
                "--checkpoints",
                str(work / "checkpoints"),
                "--previous",
                str(work),
            ],
            cwd=target,
            check=True,
        )
    for command in [
        ["verify_inputs.py", "--model", str(model)],
        ["analyze.py"],
        ["verify.py"],
        ["diagnose.py"],
        ["report.py"],
    ]:
        subprocess.run([sys.executable, *command], cwd=target, check=True)


if __name__ == "__main__":
    main()
