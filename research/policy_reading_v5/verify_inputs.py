"""Run frozen input tests using a caller-provided local tokenizer directory."""

import argparse
import unittest
import test_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    test_inputs.DEFAULT_MODEL = args.model
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_inputs)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
