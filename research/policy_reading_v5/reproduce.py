"""Re-fit nine new models in a fresh directory; preserve public legacy predictions."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from common import ROOT, V3, V4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    model = args.model.resolve()
    assert not work.exists(), "Use a fresh output directory"
    assert model.is_dir()
    work.mkdir(parents=True)
    for dependency in [V3, V4]:
        shutil.copytree(
            dependency,
            work / dependency.name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    target = work / ROOT.name
    target.mkdir()
    for source in ROOT.glob("*.py"):
        shutil.copy2(source, target / source.name)
    for name in [
        "protocol.json",
        "freeze.json",
        "input_audit.json",
        "routing_protocol.json",
        "routing_freeze.json",
        "routing_summary.json",
    ]:
        shutil.copy2(ROOT / name, target / name)
    shutil.copytree(ROOT / "data", target / "data")
    for condition in ["legacy_plain", "legacy_marked"]:
        shutil.copytree(ROOT / "results" / condition, target / "results" / condition)
    shutil.copytree(ROOT / "routing_results", target / "routing_results")
    if args.prepare_only:
        print("Prepared", target)
        return
    for condition in ["plain", "marked", "blind"]:
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
        ["verify_routing.py"],
        ["report.py"],
    ]:
        subprocess.run([sys.executable, *command], cwd=target, check=True)


if __name__ == "__main__":
    main()
