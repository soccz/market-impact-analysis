"""Run a fresh copy without overwriting the published experiment."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--regime", choices=["initial", "fit", "units", "scope", "both"], default="fit"
    )
    args = p.parse_args()
    assert not args.work.exists(), "Choose a fresh work directory"
    assert Path(args.model).exists(), "Prepare the public model first"
    args.work.mkdir(parents=True)
    for item in ROOT.iterdir():
        if item.name in [
            "results",
            "results_fit",
            "results_augmentation",
            "figures",
            "__pycache__",
            ".git",
            "checkpoints",
        ]:
            continue
        target = args.work / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    if args.prepare_only:
        return

    def run(*command):
        subprocess.run([sys.executable, *command], cwd=args.work, check=True)

    if args.regime in ["units", "scope", "both"]:
        run(
            "train_augmentation.py",
            "--condition",
            args.regime,
            "--model",
            args.model,
            "--checkpoints",
            str(args.work / "checkpoints"),
        )
        run("evaluate_augmentation.py", "--condition", args.regime)
        run("verify.py", "--augmentation", args.regime)
        return
    run("baseline.py")
    if args.regime == "fit":
        shutil.copytree(
            args.work / "results/character", args.work / "results_fit/character"
        )
    run(
        "train_fit.py" if args.regime == "fit" else "train_partial.py",
        "--model",
        args.model,
        "--checkpoints",
        str(args.work / "checkpoints"),
    )
    run("evaluate_fit.py" if args.regime == "fit" else "evaluate.py")
    run("verify.py", *(["--fit"] if args.regime == "fit" else []))


if __name__ == "__main__":
    main()
