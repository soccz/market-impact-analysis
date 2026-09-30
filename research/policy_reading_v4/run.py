"""Six new fits or reuse of six v3 models; input transformation is the intervention."""

import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from common import ROOT, V3, datasets, check_lock, sha, save
from model import DEFAULT_MODEL, loss
from partial_model import Reader
from representation import prepare, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        required=True,
        choices=["raw", "raw_scope", "normalized", "normalized_scope"],
    )
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    check_lock()
    config = json.loads((ROOT / "protocol.json").read_text())["training"]
    normalized = args.condition.startswith("normalized")
    scope = args.condition.endswith("scope")
    sets = datasets(scope)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    encoded = {
        name: prepare(rows, tokenizer, normalized, config["max_length"])
        for name, rows in sets.items()
    }
    train = encoded["train"]["encoded"][0]
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for seed in config["seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        folder = ROOT / "results" / args.condition / str(seed)
        checkpoint = args.checkpoints / args.condition / str(seed)
        checkpoint.mkdir(parents=True, exist_ok=True)
        if not args.predict_only:
            assert not folder.exists(), f"Refusing overwrite: {folder}"
            folder.mkdir(parents=True)
        model = Reader(args.model, "evidence").to("cuda")
        if args.predict_only or not normalized:
            if normalized:
                weight = checkpoint / "model.pt"
                expected = json.loads((folder / "training.json").read_text())[
                    "checkpoint_sha256"
                ]
            else:
                sub = "augmentation_scope" if scope else "fit_checkpoints"
                weight = args.previous / sub / f"evidence-{seed}/model.pt"
                log = (
                    V3
                    / ("results_augmentation/scope" if scope else "results_fit")
                    / f"evidence-{seed}/training.json"
                )
                expected = json.loads(log.read_text())["checkpoint_sha256"]
            assert sha(weight) == expected
            model.load_state_dict(
                torch.load(weight, map_location="cuda", weights_only=True)
            )
            if not args.predict_only:
                save(
                    folder / "training.json",
                    dict(
                        reused=True,
                        checkpoint_sha256=expected,
                        source=str(log.relative_to(V3)),
                    ),
                )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if p.requires_grad and n.startswith("encoder.")
                        ],
                        "lr": config["lr"],
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if p.requires_grad and not n.startswith("encoder.")
                        ],
                        "lr": config["head_lr"],
                    },
                ],
                weight_decay=config["weight_decay"],
            )
            steps = (
                math.ceil(len(sets["train"]) / config["effective_batch"])
                * config["epochs"]
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(steps * config["warmup_fraction"]), steps
            )
            scaler = torch.cuda.amp.GradScaler()
            rng = torch.Generator().manual_seed(seed)
            log = []
            started = time.time()
            for epoch in range(config["epochs"]):
                model.train()
                order = torch.randperm(len(sets["train"]), generator=rng)
                for group in range(0, len(order), config["effective_batch"]):
                    optimizer.zero_grad(set_to_none=True)
                    group_ids = order[group : group + config["effective_batch"]]
                    batch_loss = 0
                    for offset in range(0, len(group_ids), config["micro_batch"]):
                        ids = group_ids[offset : offset + config["micro_batch"]]
                        batch = {k: v[ids].to("cuda") for k, v in train.items()}
                        with torch.cuda.amp.autocast():
                            output = model(
                                **{
                                    k: batch[k]
                                    for k in [
                                        "input_ids",
                                        "attention_mask",
                                        "token_type_ids",
                                    ]
                                }
                            )
                            value = (
                                loss(output, batch, "evidence")
                                * len(ids)
                                / len(group_ids)
                            )
                        assert torch.isfinite(value)
                        scaler.scale(value).backward()
                        batch_loss += float(value.detach())
                    scaler.unscale_(optimizer)
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad), 1.0
                        )
                    )
                    oldscale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = scaler.get_scale() < oldscale
                    if not skipped:
                        scheduler.step()
                    log.append(
                        dict(
                            step=len(log) + 1,
                            epoch=epoch + 1,
                            loss=batch_loss,
                            grad_norm=norm if math.isfinite(norm) else str(norm),
                            skipped=skipped,
                        )
                    )
                print(
                    args.condition,
                    seed,
                    "epoch",
                    epoch + 1,
                    "loss",
                    round(batch_loss, 4),
                    "seconds",
                    round(time.time() - started),
                    flush=True,
                )
            torch.save(model.cpu().state_dict(), checkpoint / "model.pt")
            model.to("cuda")
            save(
                folder / "training.json",
                dict(
                    reused=False,
                    seed=seed,
                    condition=args.condition,
                    seconds=time.time() - started,
                    steps=log,
                    checkpoint_sha256=sha(checkpoint / "model.pt"),
                    environment=dict(
                        torch=torch.__version__,
                        cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(),
                    ),
                ),
            )
        for name, rows in sets.items():
            target = (
                checkpoint / f"reproduced_{name}.json"
                if args.predict_only
                else folder / f"{name}.json"
            )
            save(target, predict(model, rows, encoded[name], "cuda"), compact=True)
        print("FINISHED", args.condition, seed, flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
