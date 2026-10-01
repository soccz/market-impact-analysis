"""Extract query-independent numeric date branches; preserve first responses."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
from core import sha
from infer import ROOT, MODELS, OPTIONS, obj, array, request

PROMPT = "주어진 정책의 보증료 지원 금액 상한만 구조화하라. 가상 신청자나 질문은 없다. 원화 단위 정수로 변환하라. 보증 가입일에 따른 기본 규칙과 이전 가입자 예외를 빠짐없이 서로 겹치지 않는 날짜 구간으로 나눈다. from_inclusive는 시작일을 포함하고 until_exclusive는 종료일을 포함하지 않는다. 무제한 시작 또는 끝은 null이다. 날짜는 YYYY-MM-DD이다. 날짜 제한이 없는 단일 상한은 양쪽 null 한 구간이다. 인쇄된 연도를 실제 가입 가능 기간으로 새로 추정하지 않는다. 근거에서 읽은 것만 구조화하고 각 분기에 입력 근거 ID를 연결하라. JSON만 출력하라."


def contract_request(evidence, model):
    nullable = {"type": ["string", "null"]}
    branch = obj(
        {
            "from_inclusive": nullable,
            "until_exclusive": nullable,
            "max_krw": {"type": "integer", "minimum": 0},
            "evidence": array(evidence),
        }
    )
    req = {
        "model": MODELS[model],
        "messages": [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": json.dumps({"evidence": evidence}, ensure_ascii=False),
            },
        ],
        "format": obj({"branches": {"type": "array", "minItems": 1, "items": branch}}),
        "options": dict(OPTIONS),
        "stream": False,
        "keep_alive": "5m",
    }
    if model == "qwen":
        req["think"] = False
    return req


def run(cache, model, base="http://127.0.0.1:11434"):
    cache = Path(cache)
    data = json.loads((cache / "curated_inputs.json").read_text())
    bundle = next(b for b in data["bundles"] if b["id"] == "seoul")
    cap = {
        p: {k: v for k, v in bundle[p].items() if k.endswith("CAP")}
        for p in ["before", "after"]
    }
    cases = [c for c in data["cases"] if c["id"].startswith("seoul-cap-")]
    model_digest = next(
        t["digest"]
        for t in requests.get(base + "/api/tags", timeout=20).json()["models"]
        if t["name"] == MODELS[model]
    )
    version = requests.get(base + "/api/version", timeout=20).json()
    jobs = [
        ("contract", phase, contract_request(cap[phase], model))
        for phase in ["before", "after"]
    ]
    jobs += [
        (
            "matched",
            c["id"],
            request(
                {"before": cap["before"], "after": cap["after"]}, c, model, "after"
            ),
        )
        for c in cases
    ]
    for stage, key, req in jobs:
        path = cache / "contract" / model / stage / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert json.loads(path.read_text())["request_sha256"] == sha(req)
            continue
        start = time.monotonic()
        r = requests.post(base + "/api/chat", json=req, timeout=360)
        r.raise_for_status()
        raw = r.json()
        try:
            parsed = json.loads(raw.get("message", {}).get("content", ""))
        except json.JSONDecodeError:
            parsed = None
        record = dict(
            id=key,
            stage=stage,
            model=model,
            model_name=MODELS[model],
            model_digest=model_digest,
            server=version,
            created_utc=datetime.now(timezone.utc).isoformat(),
            request_sha256=sha(req),
            input_sha256=sha(req["messages"]),
            response=raw,
            parsed=parsed,
            elapsed_seconds=round(time.monotonic() - start, 3),
        )
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(model, stage, key, flush=True)
