"""Check saved first-response provenance against the exact frozen request builders."""

from datetime import datetime
import json
from pathlib import Path

from extension_infer import make_request
from infer import ROOT, MODELS, OPTIONS, SCHEMA, digest, messages
from scoped_infer import SCOPED_SCHEMA
from thinking_infer import request_for as thinking_request
from sampling_infer import request_for as sampling_request
from citation_infer import make_request as citation_request


def audit(cache):
    cache = Path(cache)
    rows = {
        k: json.loads(
            (
                ROOT / (k + ".json")
                if k in ["development", "composition"]
                else cache / (k + "_inputs.json")
            ).read_text()
        )
        for k in ["development", "composition", "official", "replication", "followup"]
    }
    model_info = json.loads((ROOT / "models.json").read_text())["models"]
    jobs = []
    for split in ["development", "composition", "official"]:
        for model in ["kanana", "qwen"]:
            for method in ["baseline", "scoped"]:
                folder = (
                    (ROOT / "results" if split != "official" else cache / "results")
                    / split
                    / (model + "-" + method)
                )
                jobs.append(("first", split, model, method, folder))
    for split in ["development", "official", "replication"]:
        for model, methods in [
            ("kanana-public", ["baseline", "evidence"]),
            (
                "qwen",
                ["baseline", "evidence"] if split == "replication" else ["evidence"],
            ),
        ]:
            for method in methods:
                folder = (
                    (
                        ROOT / "results/extension"
                        if split == "development"
                        else cache / "results/extension"
                    )
                    / split
                    / (model + "-" + method)
                )
                jobs.append(("extension", split, model, method, folder))
    for split in ["replication", "followup"]:
        for method in ["budget-control", "thinking"]:
            jobs.append(
                (
                    "thinking",
                    split,
                    "qwen",
                    method,
                    cache / "results/thinking" / split / ("qwen-" + method),
                )
            )
    for method in ["sampled-control", "sampled-thinking"]:
        jobs.append(
            (
                "sampling",
                "followup",
                "qwen",
                method,
                cache / "results/sampling/followup" / ("qwen-" + method),
            )
        )
    for split in ["official", "replication"]:
        for model in ["kanana-public", "qwen"]:
            jobs.append(
                (
                    "citation",
                    split,
                    model,
                    "citation-only",
                    cache / "results/citation" / split / (model + "-citation-only"),
                )
            )
    count = 0
    hashes = set()
    models = {}
    for stage, split, model, method, folder in jobs:
        for row in rows[split]:
            record = json.loads((folder / (row["id"] + ".json")).read_text())
            if stage == "first":
                prompt = (
                    json.loads((ROOT / "prompts.json").read_text())["baseline"]
                    if method == "baseline"
                    else (ROOT / "scoped_prompt.txt").read_text()
                )
                request = dict(
                    model=MODELS[model],
                    messages=messages(row, prompt),
                    stream=False,
                    format=SCHEMA if method == "baseline" else SCOPED_SCHEMA,
                    options=OPTIONS,
                    keep_alive="5m",
                )
                if model == "qwen":
                    request["think"] = False
            elif stage == "extension":
                request = make_request(row, model, method)
            elif stage == "thinking":
                request = thinking_request(row, method == "thinking")
            elif stage == "citation":
                request = citation_request(row, model, method)
            else:
                request = sampling_request(row, method == "sampled-thinking")
            rh = digest(
                json.dumps(request, ensure_ascii=False, sort_keys=True).encode()
            )
            assert rh == record["request_sha256"], (
                stage,
                split,
                model,
                method,
                row["id"],
            )
            assert record["model_digest"] == model_info[model]["manifest_digest"], model
            ih = digest(
                json.dumps(
                    {"sentences": row["sentences"], "claim": row["claim"]},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            )
            assert ih == record["input_sha256"]
            try:
                parsed = json.loads(
                    record["response"].get("message", {}).get("content", "")
                )
            except json.JSONDecodeError:
                parsed = None
            assert parsed == record["parsed"]
            freeze = (
                (
                    "development_freeze.json"
                    if split == "development"
                    else "test_freeze.json"
                )
                if stage == "first"
                else (
                    "extension_method_freeze.json"
                    if stage == "extension"
                    else (
                        "thinking_method_freeze.json"
                        if stage == "thinking"
                        else (
                            "citation_freeze.json"
                            if stage == "citation"
                            else "sampling_freeze.json"
                        )
                    )
                )
            )
            f = json.loads((ROOT / freeze).read_text())
            when = f.get("recorded_utc", f.get("created_utc"))
            assert datetime.fromisoformat(
                record["created_utc"]
            ) > datetime.fromisoformat(when)
            if split in {"replication", "followup"}:
                ref = json.loads((ROOT / (split + "_freeze.json")).read_text())
                assert datetime.fromisoformat(
                    record["created_utc"]
                ) > datetime.fromisoformat(ref["created_utc"])
            hashes.add(rh)
            count += 1
            models[model] = models.get(model, 0) + 1
    return dict(
        first_responses=count,
        unique_requests=len(hashes),
        model_response_counts=models,
        all_request_and_input_hashes_match=True,
        freeze_order_verified=True,
        chronology_scope="Development data precedes development responses; scoped development is method development, not a held-out method test. First official/composition methods and inputs precede responses. Extension methods precede responses; new reference freezes precede replication/follow-up responses.",
    )
