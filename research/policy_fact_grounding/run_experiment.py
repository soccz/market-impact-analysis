import argparse
from pathlib import Path
from bridge import run, MODELS
from report import export

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--dest", required=True)
    a = p.parse_args()
    for m in MODELS:
        run(a.inputs, str(Path(a.cache) / m), m)
    export(a.inputs, a.cache, a.dest)
