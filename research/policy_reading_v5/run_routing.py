"""Infer from the same three retained v4 scope weights after freezing retrieval."""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from common import ROOT, V4, datasets, check_lock, sha, save
from model import DEFAULT_MODEL
from partial_model import Reader
from routing import prepare, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeat-output", type=Path)
    args = parser.parse_args()
    check_lock()
    lock = json.loads((ROOT / "routing_freeze.json").read_text())
    for rel, digest in lock["files"].items():
        assert sha(ROOT / rel) == digest, rel
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    sets = datasets()
    encoded = {n: prepare(r, tokenizer) for n, r in sets.items()}
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for seed in [17, 42, 2026]:
        path = args.previous / "normalized_scope" / str(seed) / "model.pt"
        log = json.loads(
            (V4 / f"results/normalized_scope/{seed}/training.json").read_text()
        )
        assert sha(path) == log["checkpoint_sha256"]
        model = Reader(args.model, "evidence").to("cuda")
        model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
        folder = (
            args.repeat_output if args.repeat_output else ROOT / "routing_results"
        ) / str(seed)
        assert not folder.exists(), "Refusing overwrite"
        for name, rows in sets.items():
            save(
                folder / (name + ".json"),
                predict(model, rows, encoded[name], "cuda"),
                compact=True,
            )
        save(
            folder / "model.json",
            dict(
                reused=True,
                sha256=sha(path),
                source=f"v4/normalized_scope/{seed}",
                training=False,
                excluded_amount_roles="other assigned by retrieval rule, not neural inference",
            ),
        )
        del model
        torch.cuda.empty_cache()
        print("ROUTED", seed, flush=True)


if __name__ == "__main__":
    main()
