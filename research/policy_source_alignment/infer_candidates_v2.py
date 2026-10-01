"""Declared post-development clarification after both models over-abstained."""

import argparse, json
from pathlib import Path
from bridge import infer, MODELS
from infer_candidates import request as initial_request

EXTRA = """
이 작업은 신청자 자격 판정이 아니라 일반 규칙 구성이다. 신청자 정보가 없는 것이 정상이다. status=ready는 원문에 따라 이 범위의 규칙을 구성할 수 있다는 뜻이다. 전체 사업의 모든 조건이 주어지지 않아도 요청한 scope만 구성하면 ready다. 최소 한 후보를 올바르게 연결할 수 있고 범위의 의미가 빠지지 않으면 ready를 사용한다. status=insufficient는 필요한 규칙을 후보로 표현할 수 없을 때만 쓴다. ready는 지원이 확정됐다는 뜻이 아니다.
후보는 허용된 비교식이다. field=검사 항목, op는 le이하/lt미만/ge이상/gt초과/eq같음/ne다름, value=비교값이다. evidence/source는 출처이고 text는 원문 위치다. 원문값의 정규화는 이미 수행되었으므로 다른 숫자를 생성하지 않는다.
별도 작성 예시 1: 후보 a0=수강시간≥7, 범위가 수강시간뿐이면 {"status":"ready","clauses":[["a0"]]}.
별도 작성 예시 2: 후보 a0=거주기간≥18, a1=소득비율≤90. 두 조건을 모두 충족해야 하면 {"status":"ready","clauses":[["a0","a1"]]}.
별도 작성 예시 3: 같은 후보 중 하나라도 충족하면 {"status":"ready","clauses":[["a0"],["a1"]]}.
제외 대상이 기초생활수급자라면 급여를 받지 않는 eq false 후보들이 통과 조건이다. '모두 받지 않는다'는 네 급여 중 하나도 받지 않는 뜻이다. 같은 항목을 중복 선택하지 않는다."""


def request(bundle, case, model, stage, feedback=None):
    req = initial_request(bundle, case, model, stage, feedback)
    req["messages"][0]["content"] += "\n" + EXTRA
    return req


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
