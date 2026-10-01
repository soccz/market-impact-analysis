"""Post-collection representation extension; source/query isolation unchanged."""

import argparse, json
from pathlib import Path
import infer
from infer_logic_examples import request as EXAMPLE_REQUEST
from infer import obj, enum, arr, SCALAR
from extended_logic import FIELDS, ALL_FIELDS, DERIVED

ROOT = Path(__file__).parent


def request(bundle, case, model, stage):
    req = EXAMPLE_REQUEST(bundle, case, model, stage)
    if stage not in ["policy", "profile"]:
        return req
    payload = json.loads(req["messages"][1]["content"])
    payload["fields"] = ALL_FIELDS if stage == "policy" else FIELDS
    req["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    if stage == "profile":
        req["format"]["properties"]["values"]["items"]["properties"]["field"] = enum(
            FIELDS
        )
        req["messages"][0][
            "content"
        ] += "\n현재/최종 학적과 이전 학교 졸업 사실을 구분한다. 이전 학교 졸업을 현재 학교 졸업으로 바꾸지 않는다. 원격/일반/학점은행제, 졸업예정 연월, 수료증명서 여부를 명시된 경우에만 추출한다."
    else:
        schema = req["format"]
        schema["properties"]["clauses"]["items"]["items"]["properties"]["field"] = enum(
            ALL_FIELDS
        )
        term = obj(
            {
                "field": enum(f for f, v in FIELDS.items() if v["type"] == "number"),
                "numerator": {"type": "number"},
                "denominator": {"type": "number"},
            }
        )
        definition = obj(
            {
                "field": enum(DERIVED),
                "constant": {"type": "number"},
                "terms": arr(term),
                "evidence": arr(enum(bundle["evidence"])),
            }
        )
        schema["properties"]["derived"] = arr(definition)
        schema["required"].append("derived")
        req["messages"][0][
            "content"
        ] += "\n선형 계산은 derived에 정의할 수 있다. derived_i = constant + 각 terms의 field * numerator / denominator 합계다. 예를 들어 X + Y*7%/12는 X의 numerator=1, denominator=1과 Y의 numerator=0.07, denominator=12를 쓰면 된다. 이 예시 계수를 복사하지 말고 실제 근거의 계산식을 읽는다. 정의한 derived_i는 clauses에서 수치 필드로 비교한다. 계산이 없으면 derived=[]. 정의를 서로 참조하거나 신청자 값을 정책에 넣지 않는다. 금액 단위는 모두 원이며 계수를 반올림하지 않는다. status=insufficient일 때 derived와 clauses 모두 빈 배열이다."
        req["options"]["num_predict"] = 3072
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
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=infer.MODELS, required=True)
    a = p.parse_args()
    run(a.inputs, a.out, a.model)
