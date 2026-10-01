"""Source-only model composition of grounded candidate IDs, with optional feedback."""

import argparse, json
from pathlib import Path
from bridge import infer, MODELS, OPTIONS, obj, arr, enum
from candidates import extract, rule_selection, compile_selection

SYSTEM = """한국어 정책의 주어진 범위에서 자격을 충족하는 규칙을 만든다. 제공된 candidates는 원문 숫자·표현에서 기계적으로 만든 후보이지 모두 동시에 적용되는 확정 정답이 아니다. 원문과 scope를 읽고 필요한 후보 ID를 선택해 논리적으로 연결한다. clauses 바깥 배열은 OR(하나라도), 안쪽 배열은 AND(모두)다. 모든 필드를 채우거나 근거 없는 조건을 만들지 않는다. 하한과 상한을 동시에 충족해야 하면 같은 내부 배열에 넣는다. 제외 조건은 통과할 때의 반대 조건 후보로 이미 표시되어 있다. 월세환산이면 보증금 단독 한도 대신 실제 환산액 한도를 사용한다. 선언되지 않은 후보를 만들 수 없다. 부족하거나 예외를 후보로 표현할 수 없으면 insufficient와 빈 clauses를 낸다. 신청자 사례의 값이나 정답은 주어지지 않는다."""


def request(bundle, case, model, stage, feedback=None):
    assert stage == "policy"
    catalog = extract(bundle)
    ids = [x["id"] for x in catalog["atoms"]]
    payload = dict(
        scope=bundle["scope"], evidence=bundle["evidence"], candidates=catalog
    )
    if feedback is not None:
        payload["mechanical_feedback"] = feedback
    req = dict(
        model=MODELS[model],
        messages=[
            dict(role="system", content=SYSTEM),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
        options=dict(OPTIONS, num_predict=3072),
        format=obj(
            dict(
                status=enum(["ready", "insufficient"]),
                clauses=arr(arr(enum(ids or ["__none__"]))),
            )
        ),
        stream=False,
        keep_alive="5m",
    )
    if model == "qwen":
        req["think"] = False
    return req


def run(inputs, out, model, feedbacks=None):
    old = infer.request
    try:
        infer.request = lambda b, c, m, s: request(
            b, c, m, s, None if feedbacks is None else feedbacks.get((m, b["id"]))
        )
        infer.run(inputs, out, model)
    finally:
        infer.request = old


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    for m in MODELS:
        run(a.inputs, str(Path(a.cache) / m), m)
