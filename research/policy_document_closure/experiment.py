"""Frozen requests and resumable local model experiment, without answer feedback."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests
from method import retrieve

ROOT = Path(__file__).parent
MODELS = {"qwen": "qwen3.5:9b", "kanana-public": "policy-kanana15-public:q4km-f7ae0cc1"}
OPTIONS = dict(
    temperature=0,
    seed=20261002,
    num_ctx=16384,
    top_k=1,
    top_p=1,
    repeat_penalty=1,
    presence_penalty=0,
)


def sha(value):
    return hashlib.sha256(
        value
        if isinstance(value, bytes)
        else json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def obj(fields):
    return dict(
        type="object",
        properties=fields,
        required=list(fields),
        additionalProperties=False,
    )


def enum(values):
    return dict(type="string", enum=list(values))


def array(item):
    return dict(type="array", items=item)


def request(document, case, model, stage):
    evidence = document["evidence"]
    if stage == "retrieved":
        selected = retrieve(document)["expanded"]
        evidence = {key: evidence[key] for key in selected}
    payload = dict(
        scope=document["scope"], fields=document["fields"], evidence=evidence
    )
    source_only = (
        "한국어 정책 독해 실험이다. 제공된 근거와 판정 범위만 읽는다. 외부 지식이나 다른 사업을 섞지 않는다. "
        "다른 주체의 조건을 신청자에게 옮기지 않는다. 언급되지 않은 조건을 만들지 않는다. "
        "fields의 단위와 자료형을 따른다. 근거에 쓰인 예외와 참조 정의도 함께 읽는다."
    )
    if stage == "direct":
        payload["facts"] = case["facts"]
        schema = obj(
            dict(
                decision=enum(["eligible", "ineligible", "undetermined"]),
                evidence=array(enum(evidence)),
            )
        )
        prompt = source_only + (
            " facts는 검증용으로 주어진 사실이며 누락/null은 미확인이다. "
            "범위 내 조건을 충족하면 eligible, 충족하지 않으면 ineligible이다. "
            "미확인 사실의 가능한 값에 따라 결론이 달라지면 undetermined이다. "
            "결론에 영향 없는 미확인은 보류 이유가 아니다. JSON만 출력한다."
        )
        budget = 256
    else:
        node = obj(
            dict(
                id=dict(type="string"),
                op=enum(["atom", "all", "any", "not", "ref"]),
                field=enum(["", *document["fields"]]),
                cmp=enum(["eq", "ne", "ge", "gt", "le", "lt"]),
                value=dict(type=["number", "boolean"]),
                children=array(dict(type="string")),
                evidence=array(enum(evidence)),
            )
        )
        schema = obj(
            dict(
                status=enum(["ready", "insufficient"]),
                root=dict(type="string"),
                nodes=array(node),
            )
        )
        prompt = source_only + (
            " 신청자와 정답은 제공하지 않는다. 정책 자체를 중첩 조건 그래프로 옮긴다. "
            "atom은 field cmp value이며 children=[]이다. all은 모두, any는 하나 이상, not은 자식 하나의 부정, "
            "ref는 정의 노드 하나의 참조이다. 모든 노드는 고유 id와 실제 근거 id를 가진다. "
            "root는 최종 충족 조건의 id다. 비atom의 field='', cmp='eq', value=0으로 쓴다. "
            "모든 노드는 root에서 도달 가능해야 하며 순환은 금지한다. "
            "원칙적 제외 X에 예외 Y가 있으면 해당 제외는 X AND NOT Y이며, 자격은 이를 부정한다. "
            "서로 다른 필수 조건을 OR로 합치지 않는다. 범위 밖 조건은 제외한다. "
            "근거가 부족하면 status=insufficient, root='', nodes=[]로 쓴다. JSON만 출력한다."
        )
        budget = 4096
    value = dict(
        model=MODELS[model],
        messages=[
            dict(role="system", content=prompt),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
        options=dict(OPTIONS, num_predict=budget),
        format=schema,
        stream=False,
        keep_alive="10m",
    )
    if model == "qwen":
        value["think"] = False
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.inputs).read_text())
    docs = {d["id"]: d for d in data["documents"]}
    base = "http://127.0.0.1:11434"
    tags = requests.get(base + "/api/tags", timeout=20).json()["models"]
    version = requests.get(base + "/api/version", timeout=20).json()
    jobs = [
        (stage, d["id"], d, None)
        for d in docs.values()
        for stage in ["full", "retrieved"]
    ]
    jobs += [("direct", c["id"], docs[c["document"]], c) for c in data["cases"]]
    for model, name in MODELS.items():
        digest = next(t["digest"] for t in tags if t["name"] == name)
        for stage, key, doc, case in jobs:
            req = request(doc, case, model, stage)
            path = Path(args.cache) / model / stage / (key + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                saved = json.loads(path.read_text())
                assert (
                    saved["request_sha256"] == sha(req)
                    and saved["model_digest"] == digest
                )
                continue
            started = time.monotonic()
            response = requests.post(base + "/api/chat", json=req, timeout=600)
            response.raise_for_status()
            raw = response.json()
            try:
                parsed = json.loads(raw.get("message", {}).get("content", ""))
            except json.JSONDecodeError:
                parsed = None
            record = dict(
                id=key,
                stage=stage,
                model=model,
                model_name=name,
                model_digest=digest,
                server=version,
                created_utc=datetime.now(timezone.utc).isoformat(),
                request_sha256=sha(req),
                input_sha256=sha(req["messages"]),
                response_sha256=sha(raw),
                response=raw,
                parsed=parsed,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            print(
                json.dumps(
                    dict(
                        model=model,
                        stage=stage,
                        id=key,
                        done=raw.get("done_reason"),
                        seconds=record["elapsed_seconds"],
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
