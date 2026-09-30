"""Nine fixed-budget runs. Private checkpoints; public raw predictions and logs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import DEFAULT_MODEL, loss, prepare, predict
from partial_model import Reader

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--only", nargs="*")
    p.add_argument("--predict-only", action="store_true")
    args = p.parse_args()
    lock = json.loads((ROOT / "partial_freeze.json").read_text())
    for name, digest in lock["files"].items():
        assert sha(ROOT / name) == digest, name
    protocol = json.loads((ROOT / "partial_protocol.json").read_text())
    cfg = protocol["training"]
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "Use the prespecified CUDA setting"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    sets = {
        x.stem: [json.loads(s) for s in x.read_text().splitlines()]
        for x in sorted((ROOT / "data").glob("*.jsonl"))
        if x.stem not in ["source_register"]
    }
    encoded = {k: prepare(v, tokenizer, cfg["max_length"]) for k, v in sets.items()}
    data = encoded["train"][0]
    device = "cuda"
    for seed in cfg["seeds"]:
        for mode in protocol["models"]:
            tag = f"{mode}-{seed}"
            if args.only and tag not in args.only:
                continue
            output = ROOT / "results" / tag
            checkpoint = args.checkpoints / tag
            checkpoint.mkdir(parents=True, exist_ok=True)
            if not args.predict_only:
                assert not output.exists(), f"Refusing overwrite: {output}"
                output.mkdir(parents=True)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = Reader(args.model, mode).to(device)
            log = []
            begun = time.time()
            if args.predict_only:
                model.load_state_dict(
                    torch.load(
                        checkpoint / "model.pt", map_location=device, weights_only=True
                    )
                )
            else:
                optimizer = torch.optim.AdamW(
                    (p for p in model.parameters() if p.requires_grad), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
                )
                nsteps = (
                    math.ceil(len(sets["train"]) / cfg["effective_batch"])
                    * cfg["epochs"]
                )
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, int(nsteps * cfg["warmup_fraction"]), nsteps
                )
                scaler = torch.cuda.amp.GradScaler()
                rng = torch.Generator().manual_seed(seed)
                step = 0
                for epoch in range(cfg["epochs"]):
                    model.train()
                    order = torch.randperm(len(sets["train"]), generator=rng)
                    for group in range(0, len(order), cfg["effective_batch"]):
                        optimizer.zero_grad(set_to_none=True)
                        batch_loss = 0
                        group_ids = order[group : group + cfg["effective_batch"]]
                        for offset in range(0, len(group_ids), cfg["micro_batch"]):
                            ids = group_ids[offset : offset + cfg["micro_batch"]]
                            batch = {k: v[ids].to(device) for k, v in data.items()}
                            with torch.cuda.amp.autocast():
                                logits = model(
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
                                    loss(logits, batch, mode)
                                    * len(ids)
                                    / len(group_ids)
                                )
                            assert torch.isfinite(value), tag
                            scaler.scale(value).backward()
                            batch_loss += float(value.detach())
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                        oldscale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        skipped = scaler.get_scale() < oldscale
                        if not skipped:
                            scheduler.step()
                        step += 1
                        log.append(
                            dict(
                                step=step,
                                epoch=epoch + 1,
                                loss=batch_loss,
                                grad_norm=float(norm),
                                skipped=skipped,
                            )
                        )
                        if step % 10 == 0:
                            print(
                                tag,
                                "step",
                                step,
                                "/",
                                nsteps,
                                "loss",
                                round(batch_loss, 4),
                                "seconds",
                                round(time.time() - begun),
                                flush=True,
                            )
                torch.save(model.cpu().state_dict(), checkpoint / "model.pt")
                model.to(device)
                save(
                    output / "training.json",
                    dict(
                        seed=seed,
                        mode=mode,
                        seconds=time.time() - begun,
                        steps=log,
                        checkpoint_sha256=sha(checkpoint / "model.pt"),
                        protocol_sha256=sha(ROOT / "partial_protocol.json"),
                        environment=dict(
                            torch=torch.__version__,
                            gpu=torch.cuda.get_device_name(),
                            cuda=torch.version.cuda,
                        ),
                    ),
                )
            for name, rows in sets.items():
                if name == "train":
                    continue
                target = output / f"{name}_predictions.json"
                if args.predict_only:
                    target = checkpoint / f"reproduced_{name}_predictions.json"
                save(target, predict(model, rows, encoded[name], device))
            print("FINISHED", tag, round(time.time() - begun), "seconds", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
