#!/usr/bin/env python3
"""
2차 PTEI 전수 추론 — fine-tuned KLUE-RoBERTa 모델 사용.

1차 (zero-shot mDeBERTa-XNLI) 대비:
- relevance: 모델이 직접 0/1/2 분류 (1차의 keyword/semantic gate는 보조 비교용으로 보존)
- stance: fine-tuned 모델 softmax(p_support / p_contradict / p_neutral)
- channel match: 기존 hybrid (keyword OR semantic) 그대로 사용
- PTEI 공식 동일: PTEI = relevance × I(channel) × p_support × specificity × novelty

사용법:
  python run_ptei_full_inference_v2.py \
      --ckpt models/ptei_finetuned_v1/best.pt \
      --input 50_전체기사_99539건.csv \
      --output data/processed/article_scores_v2.csv
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from ptei_utils import (
    CHANNELS,
    channel_keyword_hit,
    load_channel_keywords,
    load_hypotheses,
    load_policy_cards,
    load_policy_keywords,
)


STANCE2ID = {"support": 0, "contradict": 1, "neutral": 2}
ID2STANCE = {v: k for k, v in STANCE2ID.items()}
MAX_LEN = 384


def load_policy_directions(path="policy_cards_v1.1.csv"):
    pn_dir = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pn_dir[r["pn"]] = r["D_i_initial"]
    return pn_dir


def format_input(row, pn_dir):
    pn = str(row.get("policy_n") or row.get("pn") or "")
    alias = row.get("policy_alias") or row.get("alias", "")
    industry = row.get("policy_industry") or row.get("industry", "")
    direction = pn_dir.get(pn, "")
    title = row.get("title", "")
    content = (row.get("content", "") or "")[:2000]
    return (
        f"[정책] {alias} · 방향: {direction} · 산업: {industry}\n"
        f"[제목] {title}\n"
        f"[본문] {content}"
    )


class CorpusDataset(Dataset):
    def __init__(self, rows, tokenizer, pn_dir, max_len=MAX_LEN):
        self.rows = rows
        self.tokenizer = tokenizer
        self.pn_dir = pn_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = format_input(row, self.pn_dir)
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "row_idx": idx,
        }


class MultiTaskRoberta(nn.Module):
    def __init__(self, model_name, num_rel=3, num_stance=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.rel_head = nn.Linear(hidden, num_rel)
        self.stance_head = nn.Linear(hidden, num_stance)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.rel_head(cls), self.stance_head(cls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/ptei_finetuned_v1/best.pt")
    parser.add_argument("--input", default="50_전체기사_99539건.csv")
    parser.add_argument("--output", default="data/processed/article_scores_v2.csv")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[setup] ckpt={args.ckpt}, device={device}, batch={args.batch}", flush=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    print(f"[setup] base model: {model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultiTaskRoberta(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pn_dir = load_policy_directions()
    channel_keywords = load_channel_keywords()
    cards = load_policy_cards()

    print(f"[data] loading {args.input}…", flush=True)
    with open(args.input, encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.DictReader(f))
    if args.limit > 0:
        all_rows = all_rows[: args.limit]
    print(f"  total rows: {len(all_rows)}", flush=True)

    ds = CorpusDataset(all_rows, tokenizer, pn_dir)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=0,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "row_idx": [b["row_idx"] for b in batch],
        },
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "news_id", "pn", "alias", "industry", "rel_day", "pub_date", "provider",
        "title",
        "rel_pred", "rel_p0", "rel_p1", "rel_p2",
        "stance_pred", "p_support", "p_contradict", "p_neutral",
        "channel_hit_C1", "channel_hit_C2", "channel_hit_C3",
        "channel_hit_C4", "channel_hit_C5", "channel_hit_C6",
        "ptei_score",
    ]

    fout = open(args.output, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(fout, fieldnames=out_cols)
    writer.writeheader()

    start = time.time()
    n_done = 0
    rel_dist = Counter()
    stance_dist = Counter()

    with torch.no_grad():
        for batch in loader:
            rel_logits, stance_logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            rel_probs = torch.softmax(rel_logits, dim=-1).cpu().numpy()
            stance_probs = torch.softmax(stance_logits, dim=-1).cpu().numpy()
            rel_pred = rel_probs.argmax(-1)
            stance_pred_raw = stance_probs.argmax(-1)
            # 룰: rel=0 → stance=neutral
            stance_pred = stance_pred_raw.copy()
            stance_pred[rel_pred == 0] = STANCE2ID["neutral"]

            for i, row_idx in enumerate(batch["row_idx"]):
                row = all_rows[row_idx]
                pn = str(row.get("policy_n", ""))
                content = row.get("content", "") or ""

                # channel match (1차와 동일 keyword 매칭)
                channel_hits = {}
                for ch in CHANNELS:
                    hit, _ = channel_keyword_hit(content, ch, channel_keywords)
                    channel_hits[ch] = 1 if hit else 0

                # PTEI = relevance × I(any_channel) × p_support × 1.0 (specificity·novelty 보류)
                rel_score = float(rel_pred[i]) / 2  # 0, 0.5, 1.0
                p_support = float(stance_probs[i][STANCE2ID["support"]])
                p_contradict = float(stance_probs[i][STANCE2ID["contradict"]])
                p_neutral = float(stance_probs[i][STANCE2ID["neutral"]])
                any_channel = 1 if any(channel_hits.values()) else 0
                # rel=0이면 PTEI=0
                if rel_pred[i] == 0:
                    p_support_effective = 0.0
                else:
                    p_support_effective = p_support
                ptei = rel_score * any_channel * p_support_effective

                writer.writerow({
                    "news_id": row.get("news_id", ""),
                    "pn": pn,
                    "alias": row.get("policy_alias", ""),
                    "industry": row.get("policy_industry", ""),
                    "rel_day": row.get("rel_day", ""),
                    "pub_date": row.get("pub_date", ""),
                    "provider": row.get("provider", ""),
                    "title": row.get("title", ""),
                    "rel_pred": int(rel_pred[i]),
                    "rel_p0": round(float(rel_probs[i][0]), 4),
                    "rel_p1": round(float(rel_probs[i][1]), 4),
                    "rel_p2": round(float(rel_probs[i][2]), 4),
                    "stance_pred": ID2STANCE[int(stance_pred[i])],
                    "p_support": round(p_support, 4),
                    "p_contradict": round(p_contradict, 4),
                    "p_neutral": round(p_neutral, 4),
                    **{f"channel_hit_{ch}": channel_hits[ch] for ch in CHANNELS},
                    "ptei_score": round(ptei, 4),
                })
                rel_dist[int(rel_pred[i])] += 1
                stance_dist[ID2STANCE[int(stance_pred[i])]] += 1
                n_done += 1

            if n_done % args.progress_every < args.batch:
                elapsed = time.time() - start
                rate = n_done / elapsed
                eta_sec = (len(all_rows) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(all_rows)}] {rate:.1f}/s, ETA {eta_sec/60:.1f}min", flush=True)

    fout.close()
    elapsed = time.time() - start
    print(f"\n[done] {n_done}건, {elapsed/60:.1f}분", flush=True)
    print(f"  relevance: {dict(sorted(rel_dist.items()))}", flush=True)
    print(f"  stance: {dict(sorted(stance_dist.items()))}", flush=True)
    print(f"  output: {args.output}", flush=True)

    # 요약 리포트
    summary_path = Path(args.output).parent / "v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_processed": n_done,
            "elapsed_minutes": round(elapsed / 60, 2),
            "rel_dist": dict(rel_dist),
            "stance_dist": dict(stance_dist),
            "ckpt": args.ckpt,
            "model_name": model_name,
        }, f, indent=2, ensure_ascii=False)
    print(f"  summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
