"""One bounded retry selected only by schema/global-logic feedback, never labels."""

import argparse, json, shutil
from pathlib import Path
import infer, evaluate, evaluate_capability
from infer_explicit import request as ordinary
from infer_extended import request as extended_request
from logic import policy
from worlds import execute
from extended_logic import compile_program, solve

ROOT = Path(__file__).parent
SPECS = {
    "official_repair": ("official", "official_reference.json", False),
    "composition_repair": ("composition", "composition.json", False),
    "capability_repair": ("capability_extended", "capability_reference.json", True),
}
PROMPT = "검증기는 원문 정답을 모른다. 이전 정책 구조에서 다음 문제를 발견했다. 원문과 scope만 다시 읽고 정책 구조 전체를 한 번 다시 출력하라. 조건 묶음·예외·제외 방향과 정의되지 않은 계산 변수를 확인하라. 원문 자체가 모순이거나 항상 참인 규칙이라면 그 의미를 유지해야 한다. 검증 경고만으로 정답을 반대로 바꾸지 않는다. 표현할 수 없는 핵심 조건은 insufficient로 둔다. 신청자 사례나 참조 정답은 제공되지 않는다."


def feedback(p, allowed, extended=False):
    if extended:
        formula, _, status = compile_program(p, allowed)
        value = solve(formula, {})[0] if formula is not None else None
    else:
        clauses, status = policy(p, allowed)
        value = execute(clauses, {})[0] if clauses is not None else None
    if status == "policy_insufficient":
        return None
    if status != "valid":
        return {"kind": "schema_or_type", "status": status}
    if value is not None:
        return {
            "kind": "global_constant",
            "condition_value": value,
            "meaning": (
                "제공된 구조 안에서 모든 신청자가 통과한다"
                if value
                else "제공된 구조 안에서 어떤 신청자도 통과할 수 없다"
            ),
        }
    return None


def builder(origin_records, extended):
    lookup = {(r["model"], r["stage"], r["id"]): r for r in origin_records}

    def request(b, c, m, stage):
        r = (extended_request if extended else ordinary)(b, c, m, stage)
        if stage == "policy":
            old = lookup[m, stage, b["id"]]["parsed"]
            why = feedback(old, b["evidence"], extended)
            if why:
                r["messages"] += [
                    {
                        "role": "assistant",
                        "content": json.dumps(old, ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": PROMPT + "\n" + json.dumps(why, ensure_ascii=False),
                    },
                ]
        return r

    return request


def selection():
    rows = []
    for split, (origin, reference, ext) in SPECS.items():
        data = json.loads((ROOT / reference).read_text())
        bs = {b["id"]: b for b in data["bundles"]}
        for r in json.loads(
            (ROOT / "results" / origin / "predictions.json").read_text()
        ):
            if r["stage"] == "policy":
                why = feedback(r["parsed"], bs[r["id"]]["evidence"], ext)
                rows.append(
                    dict(
                        split=split,
                        origin=origin,
                        model=r["model"],
                        id=r["id"],
                        selected=why is not None,
                        feedback=why,
                        origin_request_sha256=r["request_sha256"],
                    )
                )
    return rows


def run(cache, split, inputs):
    cache = Path(cache)
    origin, _, ext = SPECS[split]
    records = json.loads((ROOT / "results" / origin / "predictions.json").read_text())
    reqfn = builder(records, ext)
    chosen = {
        (r["model"], r["id"])
        for r in selection()
        if r["split"] == split and r["selected"]
    }
    for m in infer.MODELS:
        for stage in ["policy", "profile", "direct"]:
            dest = cache / split / m / stage
            dest.mkdir(parents=True, exist_ok=True)
            for src in (cache / origin / m / stage).glob("*.json"):
                if stage == "policy" and (m, src.stem) in chosen:
                    continue
                if not (dest / src.name).exists():
                    shutil.copy2(src, dest / src.name)
        old = infer.request
        try:
            infer.request = reqfn
            infer.run(inputs, cache / split / m, m)
        finally:
            infer.request = old
    if ext:
        old = evaluate_capability.extended_request
        try:
            evaluate_capability.extended_request = reqfn
            evaluate_capability.run(
                inputs, cache / split, ROOT / "results" / split, "extended"
            )
        finally:
            evaluate_capability.extended_request = old
    else:
        old = evaluate.request
        try:
            evaluate.request = reqfn
            evaluate.run(inputs, cache / split, ROOT / "results" / split)
        finally:
            evaluate.request = old


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    p.add_argument("--split", choices=SPECS)
    p.add_argument("--inputs")
    p.add_argument("--select", action="store_true")
    a = p.parse_args()
    if a.select:
        rows = selection()
        (ROOT / "repair_selection.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps([r for r in rows if r["selected"]], ensure_ascii=False))
    else:
        run(a.cache, a.split, a.inputs)
