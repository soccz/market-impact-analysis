"""Post-feedback representation test: explicit ALL/ANY instead of nested arrays."""

import argparse, json
from pathlib import Path
from bridge import infer, MODELS, OPTIONS, obj, arr, enum
from candidates import extract, compile_selection

PROMPT = """주어진 한국어 정책의 scope 안에서 통과 조건을 구성하라. 신청자 사례가 없는 것은 정상이며 일반 규칙을 만드는 작업이다. candidates의 원문 근거와 정규화한 조건만 사용한다.
출력 operator=ALL이면 선택한 조건을 모두 충족해야 한다. operator=ANY이면 하나라도 충족하면 된다. 원문이 '및/과/와/동시/모두'로 조건들을 요구하거나 동일 항목의 하한과 상한을 정하면 ALL이다. '하나라도/어느 하나'를 충족하면 된다고 명시하면 ANY다. 문장이 여러 개라는 이유만으로 ANY를 쓰지 않는다. 선택 scope에서 필요한 조건 ID를 conditions에 중복 없이 넣는다.
제외 조건의 후보는 이미 통과 방향으로 뒤집혀 있다. 예: 수급자 제외의 후보가 급여=false면 그 조건도 반드시 충족해야 한다. '보증금 없는 월세 가능'은 다른 금액 상한을 면제하지 않는다. '월세 없는 전세 불가'는 월세>0이 추가로 필요하다는 뜻이다. 금액 기준과 수급자 제외가 같이 적용되면 ALL이다.
환산액 상한은 derived 후보를 선택한다. 보증금 단독 상한과 환산액 상한 중 하나만 만족하면 된다는 뜻으로 ANY를 쓰지 않는다. 숫자와 %는 이미 정확 단위로 변환했으므로 새 값을 생성하지 않는다.
요청한 scope의 규칙을 표현할 수 있으면 status=ready다. 전체 사업의 다른 조건이나 신청자의 개인정보가 없다는 이유로 insufficient를 쓰지 않는다. 필요한 후보가 없거나 ALL/ANY 한 단계로 논리를 표현할 수 없으면 status=insufficient,conditions=[]로 둔다. 형식은 {"status":"ready","operator":"ALL","conditions":["a0","a1"]} 같은 JSON이다."""


def request(bundle, case, model, stage, feedback=None):
    assert stage == "policy"
    catalog = extract(bundle)
    req = dict(
        model=MODELS[model],
        messages=[
            dict(role="system", content=PROMPT),
            dict(
                role="user",
                content=json.dumps(
                    dict(
                        scope=bundle["scope"],
                        evidence=bundle["evidence"],
                        candidates=catalog,
                    ),
                    ensure_ascii=False,
                ),
            ),
        ],
        options=dict(OPTIONS, num_predict=3072),
        format=obj(
            dict(
                status=enum(["ready", "insufficient"]),
                operator=enum(["ALL", "ANY"]),
                conditions=arr(
                    enum([x["id"] for x in catalog["atoms"]] or ["__none__"])
                ),
            )
        ),
        stream=False,
        keep_alive="5m",
    )
    if model == "qwen":
        req["think"] = False
    return req


def as_selection(value):
    if (
        not isinstance(value, dict)
        or value.get("status") != "ready"
        or value.get("operator") not in ["ALL", "ANY"]
        or not isinstance(value.get("conditions"), list)
    ):
        return dict(status="insufficient", clauses=[])
    ids = value["conditions"]
    return dict(
        status="ready",
        clauses=[ids] if value["operator"] == "ALL" else [[x] for x in ids],
    )


def run(inputs, out, model):
    old = infer.request
    try:
        infer.request = request
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
