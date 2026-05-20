#!/usr/bin/env python3
"""
KLUE-RoBERTa fine-tune (멀티태스크: relevance 0/1/2 + stance support/contradict/neutral).

사용법:
  python finetune_klue_roberta.py --mode sanity     # 빠른 검증 (50건, 2epoch)
  python finetune_klue_roberta.py --mode full       # 전체 학습 (Train500, 5epoch)
  python finetune_klue_roberta.py --mode full --base # base 모델 사용 (large 대신, 빠르고 가벼움)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ---------- 설정 ----------
STANCE2ID = {"support": 0, "contradict": 1, "neutral": 2}
ID2STANCE = {v: k for k, v in STANCE2ID.items()}
MAX_LEN = 384  # MPS 메모리 고려 — 384가 안전, large는 부담 시 base로 fallback
TRAIN_CSV = "splits/train_500_final.csv"
DEV_CSV = "splits/dev100_consensus_labels.csv"
POLICY_CSV = "policy_cards_v1.1.csv"
SANITY_TRAIN_N = 50
SANITY_DEV_N = 10
SANITY_EPOCHS = 2
FULL_EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


def load_policy_directions(path):
    pn_dir = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pn_dir[r["pn"]] = r["D_i_initial"]
    return pn_dir


def format_input(row, pn_dir):
    """기사 + 정책 컨텍스트를 단일 문장으로 결합."""
    pn = row["pn"]
    alias = row.get("alias", "")
    industry = row.get("industry", "")
    direction = row.get("direction") or pn_dir.get(pn, "")
    title = row.get("title", "")
    content = row.get("content", "")[:2000]  # 토큰화 전 1차 컷
    return (
        f"[정책] {alias} · 방향: {direction} · 산업: {industry}\n"
        f"[제목] {title}\n"
        f"[본문] {content}"
    )


def load_dataset(csv_path, pn_dir, label_source="primary"):
    """
    label_source:
      - 'primary': train_500_final.csv의 relevance/stance_label (저자 검토 반영본)
      - 'consensus': dev100의 consensus_relevance/consensus_stance
    """
    examples = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if label_source == "consensus":
                rel = r.get("consensus_relevance")
                stance = r.get("consensus_stance")
            else:
                rel = r.get("relevance")
                stance = r.get("stance_label")
            if rel is None or rel == "":
                continue
            if stance is None or stance == "":
                continue
            try:
                rel_int = int(rel)
            except ValueError:
                continue
            if stance not in STANCE2ID:
                continue
            text = format_input(r, pn_dir)
            examples.append({
                "text": text,
                "rel": rel_int,
                "stance": STANCE2ID[stance],
                "news_id": r["news_id"],
            })
    return examples


class PolicyDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "rel": torch.tensor(ex["rel"], dtype=torch.long),
            "stance": torch.tensor(ex["stance"], dtype=torch.long),
            "news_id": ex["news_id"],
        }


class MultiTaskRoberta(nn.Module):
    def __init__(self, model_name, num_rel=3, num_stance=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rel_head = nn.Linear(hidden, num_rel)
        self.stance_head = nn.Linear(hidden, num_stance)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.rel_head(cls), self.stance_head(cls)


def collate_keep_ids(batch):
    """news_id는 텐서가 아니므로 분리."""
    news_ids = [b.pop("news_id") for b in batch]
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    out["news_ids"] = news_ids
    return out


def evaluate(model, loader, device):
    model.eval()
    rel_preds, rel_trues = [], []
    stance_preds, stance_trues = [], []
    with torch.no_grad():
        for batch in loader:
            rel_logits, stance_logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            rp = rel_logits.argmax(-1).cpu().numpy()
            sp = stance_logits.argmax(-1).cpu().numpy()
            # 강제 룰: rel=0 → stance=neutral
            sp[rp == 0] = STANCE2ID["neutral"]
            rel_preds.extend(rp.tolist())
            rel_trues.extend(batch["rel"].numpy().tolist())
            stance_preds.extend(sp.tolist())
            stance_trues.extend(batch["stance"].numpy().tolist())

    metrics = {
        "rel_accuracy": float(accuracy_score(rel_trues, rel_preds)),
        "rel_kappa": float(cohen_kappa_score(rel_trues, rel_preds)),
        "rel_macro_f1": float(f1_score(rel_trues, rel_preds, average="macro", zero_division=0)),
        "stance_accuracy": float(accuracy_score(stance_trues, stance_preds)),
        "stance_kappa": float(cohen_kappa_score(stance_trues, stance_preds)),
        "stance_macro_f1": float(f1_score(stance_trues, stance_preds, average="macro", zero_division=0)),
        "rel_pred_dist": dict(Counter(rel_preds)),
        "stance_pred_dist": dict(Counter([ID2STANCE[s] for s in stance_preds])),
    }
    # stance(rel>0) — 1차 추론과 비교 가능한 메트릭
    rel_pos_idx = [i for i, r in enumerate(rel_trues) if r > 0]
    if rel_pos_idx:
        sp_pos = [stance_preds[i] for i in rel_pos_idx]
        st_pos = [stance_trues[i] for i in rel_pos_idx]
        metrics["stance_kappa_rel_pos"] = float(cohen_kappa_score(st_pos, sp_pos))
    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, device, log_prefix=""):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        rel_logits, stance_logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        loss = loss_fn(rel_logits, batch["rel"].to(device)) + \
               loss_fn(stance_logits, batch["stance"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity")
    parser.add_argument("--base", action="store_true", help="klue/roberta-base 사용 (large 대신)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=0, help="override default epochs (0 = use defaults)")
    parser.add_argument("--output", default="models/ptei_finetuned_v1")
    parser.add_argument("--report", default="reports/finetune_eval_v1.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = "klue/roberta-base" if args.base else "klue/roberta-large"
    epochs = args.epochs if args.epochs > 0 else (SANITY_EPOCHS if args.mode == "sanity" else FULL_EPOCHS)
    output_dir = Path(args.output + ("_sanity" if args.mode == "sanity" else ""))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[setup] model={model_name}, device={device}, mode={args.mode}, batch={args.batch}, epochs={epochs}")

    pn_dir = load_policy_directions(POLICY_CSV)

    print("[data] loading…")
    train_data = load_dataset(TRAIN_CSV, pn_dir, label_source="primary")
    dev_data = load_dataset(DEV_CSV, pn_dir, label_source="consensus")
    print(f"  train: {len(train_data)}, dev: {len(dev_data)}")

    if args.mode == "sanity":
        rng = np.random.default_rng(args.seed)
        train_data = list(rng.choice(train_data, size=min(SANITY_TRAIN_N, len(train_data)), replace=False))
        dev_data = list(rng.choice(dev_data, size=min(SANITY_DEV_N, len(dev_data)), replace=False))
        print(f"  sanity train: {len(train_data)}, sanity dev: {len(dev_data)}")

    print("[tokenizer] loading…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = PolicyDataset(train_data, tokenizer)
    dev_ds = PolicyDataset(dev_data, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate_keep_ids, num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch, shuffle=False,
        collate_fn=collate_keep_ids, num_workers=0,
    )

    print("[model] loading…")
    model = MultiTaskRoberta(model_name).to(device)

    total_steps = len(train_loader) * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    print(f"[train] starting {epochs} epochs, {total_steps} steps total")
    best_kappa = -1.0
    history = []
    for ep in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        metrics = evaluate(model, dev_loader, device)
        snap = {"epoch": ep, "train_loss": round(avg_loss, 4)}
        snap.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})
        history.append(snap)
        print(f"  ep{ep}: loss={avg_loss:.4f} | "
              f"rel(κ={metrics['rel_kappa']:.3f}, acc={metrics['rel_accuracy']:.3f}, F1={metrics['rel_macro_f1']:.3f}) "
              f"| stance(κ={metrics['stance_kappa']:.3f}, acc={metrics['stance_accuracy']:.3f}, F1={metrics['stance_macro_f1']:.3f})")

        combined = (metrics["rel_kappa"] + metrics["stance_kappa"]) / 2
        if combined > best_kappa:
            best_kappa = combined
            ckpt = output_dir / "best.pt"
            torch.save({"model_state": model.state_dict(),
                        "model_name": model_name,
                        "epoch": ep,
                        "metrics": metrics}, ckpt)
            print(f"    ✓ saved best to {ckpt} (mean κ={combined:.3f})")

    # 리포트 저장
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": args.batch,
        "train_size": len(train_data),
        "dev_size": len(dev_data),
        "history": history,
        "best_combined_kappa": best_kappa,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[done] best mean κ = {best_kappa:.3f}, report: {args.report}")


if __name__ == "__main__":
    main()
