"""Apply frozen evaluator to the separately recorded logic-example requests."""

import argparse
import evaluate
from infer_logic_examples import request


def run(inputs, cache, dest):
    old = evaluate.request
    try:
        evaluate.request = request
        evaluate.run(inputs, cache, dest)
    finally:
        evaluate.request = old


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--dest", required=True)
    a = p.parse_args()
    run(a.inputs, a.cache, a.dest)
