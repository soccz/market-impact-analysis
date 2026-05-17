import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

from ptei_utils import (
    CHANNELS,
    channel_keyword_hit,
    load_channel_keywords,
    load_csv,
    load_hypotheses,
    load_policy_cards,
    load_policy_keywords,
    policy_key,
    relevance_v3_1,
    write_csv,
)


DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def parse_device(value):
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return torch.device(text)


def load_embedder(model_name, device):
    print(f"Loading semantic gate encoder: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if isinstance(device, torch.device):
        model.to(device)
    model.eval()

    def embed_texts(texts, batch_size):
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = [text if text and text.strip() else " " for text in texts[start : start + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
            if isinstance(device, torch.device):
                enc = {key: value.to(device) for key, value in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            vectors.extend((summed / counts).cpu().numpy())
        return vectors

    return embed_texts


def channel_seed_embeddings(channel_keywords, embed_texts, batch_size):
    embeddings = {}
    for channel, keywords in channel_keywords.items():
        sample = keywords[:20] or [channel]
        vectors = np.stack(embed_texts(sample, batch_size))
        embeddings[channel] = vectors.mean(axis=0)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="50_전체기사_99539건.csv")
    parser.add_argument("--output", default="data/processed/article_scores_hybrid.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="-1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tau-rel", type=float, default=0.4)
    parser.add_argument("--gate-mode", choices=["keyword", "hybrid"], default="hybrid")
    parser.add_argument("--embedding-model", default="klue/roberta-base")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--tau-sem", type=float, default=0.55)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    started = time.time()
    policies = load_policy_keywords()
    hypotheses = load_hypotheses()
    cards = load_policy_cards()
    channel_keywords = load_channel_keywords()

    rows = load_csv(args.input)
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded rows: {len(rows)}", flush=True)

    nli_device = parse_device(args.device)
    embedding_device = parse_device(args.embedding_device)
    embed_texts = None
    channel_embeddings = {}
    if args.gate_mode == "hybrid":
        embed_texts = load_embedder(args.embedding_model, embedding_device)
        channel_embeddings = channel_seed_embeddings(channel_keywords, embed_texts, args.embed_batch_size)

    outputs = []
    texts = []
    embed_jobs = []
    meta = []

    print("[1/4] relevance + keyword gates", flush=True)
    for idx, row in enumerate(rows):
        pn = policy_key(row.get("policy_n"))
        title = row.get("title", "")
        content = row.get("content", "")
        text = f"{title}\n{content}"
        rel_score, rel_detail = relevance_v3_1(title, content, policies[pn])
        specificity = 1.0 + float(rel_detail.get("specificity_boost", 0.0))

        out = dict(row)
        out.update(
            {
                "pn": pn,
                "rel_score_v3_1": f"{rel_score:.3f}",
                "rel_level_v3_1": rel_detail["level"],
                "rel_hit_v3_1": rel_detail.get("hit", ""),
                "specificity": f"{specificity:.3f}",
            }
        )
        outputs.append(out)
        texts.append(text)

        row_meta = {
            "pn": pn,
            "rel_score": rel_score,
            "specificity": specificity,
            "channel_hits": {},
            "channel_cosines": {},
            "active_channels": [],
            "stance_scores": {"support": 0.0, "contradict": 0.0, "neutral": 1.0},
            "pred_stance": "neutral",
        }

        if rel_score >= args.tau_rel and rel_detail.get("policy_keyword_hit", False):
            active_card_channels = cards[pn]["active_channels"]
            for channel in active_card_channels:
                hit, keyword = channel_keyword_hit(text, channel, channel_keywords)
                row_meta["channel_hits"][channel] = keyword
                if hit:
                    row_meta["active_channels"].append(channel)
            if args.gate_mode == "hybrid":
                embed_jobs.append(idx)

        meta.append(row_meta)
        if (idx + 1) % args.progress_every == 0:
            print(f"  prepared {idx + 1}/{len(rows)}", flush=True)

    if args.gate_mode == "hybrid" and embed_jobs:
        print(f"[2/4] semantic channel gates: {len(embed_jobs)} candidate rows", flush=True)
        vectors = embed_texts([texts[idx] for idx in embed_jobs], args.embed_batch_size)
        for idx, vector in zip(embed_jobs, vectors):
            pn = meta[idx]["pn"]
            for channel in cards[pn]["active_channels"]:
                sem_cos = cosine(vector, channel_embeddings[channel])
                meta[idx]["channel_cosines"][channel] = sem_cos
                if sem_cos >= args.tau_sem and channel not in meta[idx]["active_channels"]:
                    meta[idx]["active_channels"].append(channel)
        print("  semantic gates done", flush=True)

    nli_jobs = defaultdict(list)
    for idx, item in enumerate(meta):
        if item["rel_score"] >= args.tau_rel and item["active_channels"]:
            nli_jobs[item["pn"]].append(idx)

    print(f"[3/4] batched NLI: {sum(len(v) for v in nli_jobs.values())} rows across {len(nli_jobs)} policies", flush=True)
    zsc = pipeline("zero-shot-classification", model=args.model, device=nli_device)
    done = 0
    for pn in sorted(nli_jobs, key=lambda value: int(value)):
        indices = nli_jobs[pn]
        candidates = [
            hypotheses[pn]["support"],
            hypotheses[pn]["contradict"],
            hypotheses[pn]["neutral"],
        ]
        policy_texts = [texts[idx] for idx in indices]
        result = zsc(policy_texts, candidates, multi_label=False, batch_size=args.nli_batch_size)
        if isinstance(result, dict):
            result = [result]
        for idx, out in zip(indices, result):
            scores = dict(zip(out["labels"], out["scores"]))
            stance_scores = {
                "support": float(scores[candidates[0]]),
                "contradict": float(scores[candidates[1]]),
                "neutral": float(scores[candidates[2]]),
            }
            meta[idx]["stance_scores"] = stance_scores
            meta[idx]["pred_stance"] = max(stance_scores, key=stance_scores.get)
        done += len(indices)
        print(f"  NLI pn={pn}: {len(indices)} rows, cumulative {done}", flush=True)

    print("[4/4] write article scores", flush=True)
    final_rows = []
    for out, item in zip(outputs, meta):
        stance_scores = item["stance_scores"]
        active_channels = sorted(set(item["active_channels"]), key=CHANNELS.index)
        out.update(
            {
                "active_channels": ",".join(active_channels),
                "pred_stance": item["pred_stance"],
                "p_support": f"{stance_scores['support']:.6f}",
                "p_contradict": f"{stance_scores['contradict']:.6f}",
                "p_neutral": f"{stance_scores['neutral']:.6f}",
            }
        )

        total_ptei = 0.0
        total_contradiction = 0.0
        for channel in CHANNELS:
            gate = 1 if channel in active_channels else 0
            ptei = item["rel_score"] * gate * stance_scores["support"] * item["specificity"]
            contradiction = item["rel_score"] * gate * stance_scores["contradict"] * item["specificity"]
            total_ptei += ptei
            total_contradiction += contradiction
            semantic_cosine = item["channel_cosines"].get(channel, 0.0)
            out[f"{channel}_gate"] = gate
            out[f"{channel}_keyword_hit"] = item["channel_hits"].get(channel, "")
            out[f"{channel}_semantic_cosine"] = f"{semantic_cosine:.6f}"
            out[f"{channel}_semantic_gate"] = int(semantic_cosine >= args.tau_sem)
            out[f"{channel}_ptei"] = f"{ptei:.6f}"
            out[f"{channel}_contradiction"] = f"{contradiction:.6f}"

        out["PTEI_total"] = f"{total_ptei:.6f}"
        out["Contradiction_total"] = f"{total_contradiction:.6f}"
        final_rows.append(out)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, final_rows, list(final_rows[0].keys()))
    elapsed = time.time() - started
    print(f"Saved: {args.output}", flush=True)
    print(f"elapsed: {elapsed:.1f}s ({len(rows) / elapsed:.2f} rows/sec)", flush=True)


if __name__ == "__main__":
    main()
