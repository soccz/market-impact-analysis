"""Shared encoder, independent factors, monetary roles, optional evidence supervision."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from build_data import RELATIONS, STATES, amounts

DEFAULT_MODEL = "/home/soccz/.cache/huggingface/hub/models--klue--roberta-base/snapshots/02f94ba5e3fcb7e2a58a390b8639b0fac974a8da"
ROLES = ["other", "before", "after"]


class Reader(nn.Module):
    def __init__(self, checkpoint, mode):
        super().__init__()
        self.mode = mode
        self.encoder = AutoModel.from_pretrained(
            checkpoint, local_files_only=True, add_pooling_layer=False
        )
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.joint = nn.Linear(h, 24)
        self.relation = nn.Linear(h, 4)
        self.state = nn.Linear(h, 6)
        self.role = nn.Linear(h, 3)
        self.evidence = nn.Linear(h, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        z = self.dropout(
            self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state
        )
        return dict(
            joint=self.joint(z[:, 0]),
            relation=self.relation(z[:, 0]),
            state=self.state(z[:, 0]),
            role=self.role(z),
            evidence=self.evidence(z),
        )


def prepare(rows, tokenizer, max_length=128):
    batch = tokenizer(
        [x["query"] for x in rows],
        [x["text"] for x in rows],
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    max_seen = max(map(len, batch["input_ids"]))
    assert max_seen <= max_length, ("No silent truncation", max_seen, max_length)
    # Explicit fixed padding after a truncation-free check.
    batch = tokenizer(
        [x["query"] for x in rows],
        [x["text"] for x in rows],
        padding="max_length",
        max_length=max_length,
        truncation=False,
        return_offsets_mapping=True,
    )
    role = np.full((len(rows), max_length), -100, dtype=np.int64)
    evidence = np.full_like(role, -100)
    masks = []
    detected = []
    for i, row in enumerate(rows):
        seq = batch.sequence_ids(i)
        mask = [
            j
            for j, (a, b) in enumerate(batch["offset_mapping"][i])
            if seq[j] == 1 and b > a
        ]
        masks.append(mask)
        detected.append(amounts(row["text"]))
        for j in mask:
            a, b = batch["offset_mapping"][i][j]
            evidence[i, j] = 0
            for sp in row["spans"]:
                if a < sp["end"] and b > sp["start"]:
                    role[i, j] = ROLES.index(sp["role"])
            for sp in row["evidence"]:
                if a < sp["end"] and b > sp["start"]:
                    evidence[i, j] = 1 if sp["kind"] == "relation" else 2
    tensors = {
        k: torch.tensor(batch[k], dtype=torch.long)
        for k in ["input_ids", "attention_mask", "token_type_ids"]
    }
    tensors["r"] = torch.tensor([RELATIONS.index(x["relation"]) for x in rows])
    tensors["s"] = torch.tensor([STATES.index(x["state"]) for x in rows])
    tensors["role_labels"] = torch.tensor(role)
    tensors["evidence_labels"] = torch.tensor(evidence)
    return tensors, batch["offset_mapping"], masks, detected


def loss(output, batch, mode):
    ce = nn.functional.cross_entropy
    axes = (
        ce(output["joint"], batch["r"] * 6 + batch["s"])
        if mode == "flat"
        else ce(output["relation"], batch["r"]) + ce(output["state"], batch["s"])
    )
    roles = ce(output["role"].reshape(-1, 3), batch["role_labels"].reshape(-1))
    evidence = (
        ce(output["evidence"].reshape(-1, 3), batch["evidence_labels"].reshape(-1))
        if mode == "evidence"
        else axes.new_zeros(())
    )
    return axes + roles + 0.5 * evidence


def merge_evidence(labels, offsets, mask):
    spans = []
    for j in mask:
        label = int(labels[j])
        a, b = offsets[j]
        if label == 0:
            continue
        kind = "relation" if label == 1 else "state"
        if spans and spans[-1]["kind"] == kind and a <= spans[-1]["end"] + 1:
            spans[-1]["end"] = b
        else:
            spans.append(dict(start=a, end=b, kind=kind))
    return spans


@torch.no_grad()
def predict(model, rows, prepared, device, batch_size=16):
    tensors, offsets, masks, detected = prepared
    model.eval()
    results = []
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        inputs = {
            k: tensors[k][start:end].to(device)
            for k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        # Evaluation always fp32; no dynamic padding or precision mismatch between pairs.
        logits = model(**inputs)
        probs = {
            k: torch.softmax(v.float(), dim=-1).cpu().numpy() for k, v in logits.items()
        }
        for k, i in enumerate(range(start, end)):
            jp = probs["joint"][k].reshape(4, 6)
            rp = jp.sum(1) if model.mode == "flat" else probs["relation"][k]
            sp = jp.sum(0) if model.mode == "flat" else probs["state"][k]
            ri, si = (
                np.unravel_index(jp.argmax(), jp.shape)
                if model.mode == "flat"
                else (rp.argmax(), sp.argmax())
            )
            out = []
            for candidate in detected[i]:
                tok = [
                    j
                    for j in masks[i]
                    if offsets[i][j][0] < candidate["end"]
                    and offsets[i][j][1] > candidate["start"]
                ]
                p = probs["role"][k, tok].mean(0)
                out.append(
                    dict(
                        candidate, role=ROLES[int(p.argmax())], probabilities=p.tolist()
                    )
                )
            ev = (
                merge_evidence(probs["evidence"][k].argmax(-1), offsets[i], masks[i])
                if model.mode == "evidence"
                else []
            )
            for e in ev:
                e["text"] = rows[i]["text"][e["start"] : e["end"]]
            results.append(
                dict(
                    id=rows[i]["id"],
                    relation=RELATIONS[int(ri)],
                    state=STATES[int(si)],
                    relation_probabilities=rp.tolist(),
                    state_probabilities=sp.tolist(),
                    joint_probabilities=(
                        jp.reshape(-1).tolist() if model.mode == "flat" else None
                    ),
                    spans=out,
                    evidence=ev,
                )
            )
    return results
