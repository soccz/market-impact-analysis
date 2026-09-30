"""Run the frozen input tests with a caller-supplied local tokenizer path."""

import argparse
import unittest
import test_representation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    test_representation.DEFAULT_MODEL = args.model
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_representation)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
