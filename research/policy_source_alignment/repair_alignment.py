"""One source-only counterexample feedback pass; never reads reference programs."""

import argparse, json, copy
from pathlib import Path
from bridge import ROOT, MODELS, infer, sha
from candidates import extract, compile_selection, rule_selection
from infer_candidates_v2 import request as clarified_request
from counterexamples import witness


def feedback(bundle, selection):
    catalog = extract(bundle)
    pred = compile_selection(selection, catalog)
    heuristic = compile_selection(rule_selection(bundle, catalog), catalog)
    if pred["status"] != "ready":
        return dict(
            reason="no_ready_policy",
            previous_selection=selection,
            message="신청자 정보 없이 일반 규칙을 구성하는 작업이다. 후보로 요청 범위를 구성할 수 있는지 원문을 다시 확인하라.",
        )
    if heuristic["status"] != "ready":
        return None
    found = witness(pred, heuristic, list(bundle["evidence"]))
    if found["status"] != "witness":
        return None
    return dict(
        reason="source_parser_disagreement",
        previous_selection=selection,
        hypothetical_profile=found["profile"],
        model_program_decision=found["predicted"],
        restricted_parser_decision=found["reference"],
        domain=found["domains"],
        message="이 가상 조건에서 두 프로그램의 결론이 다르다. 제한 문법 해석기는 정답표가 아니므로 원문과 scope로 검토하라. 안쪽 배열은 모두 충족 AND이고 바깥 배열은 대안 OR이다. 월세 없는 계약 제외·소득 상하한·급여 제외·환산액 조건이 다른 경로 때문에 무시되지 않는지 확인하라. 원문이 명시한 대안만 OR로 둔다.",
    )


def request(bundle, case, model, stage, feedback_value=None):
    return clarified_request(bundle, case, model, stage, feedback_value)


def run(data, source_cache, dest, record_path):
    source_cache, dest = Path(source_cache), Path(dest)
    selection = []
    for model in MODELS:
        for b in data["bundles"]:
            raw = json.loads(
                (source_cache / model / "policy" / (b["id"] + ".json")).read_text()
            )
            f = feedback(b, raw["parsed"])
            selection.append(
                dict(
                    model=model,
                    bundle=b["id"],
                    selected=f is not None,
                    feedback=f,
                    prior_request_sha256=raw["request_sha256"],
                    prior_response_sha256=sha(raw["response"]),
                )
            )
    Path(record_path).write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    )
    for model in MODELS:
        selected = [r for r in selection if r["model"] == model and r["selected"]]
        chosen = {r["bundle"]: r["feedback"] for r in selected}
        for b in data["bundles"]:
            if b["id"] in chosen:
                continue
            path = dest / model / "policy" / (b["id"] + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            original = source_cache / model / "policy" / (b["id"] + ".json")
            if path.exists():
                assert path.read_bytes() == original.read_bytes()
            else:
                path.write_bytes(original.read_bytes())
        if not selected:
            continue
        inputs = dest / (model + "-selected-inputs.json")
        inputs.parent.mkdir(parents=True, exist_ok=True)
        inputs.write_text(
            json.dumps(
                dict(
                    bundles=[b for b in data["bundles"] if b["id"] in chosen], cases=[]
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        old = infer.request
        try:
            infer.request = lambda b, c, m, s: request(b, c, m, s, chosen[b["id"]])
            infer.run(inputs, dest / model, model)
        finally:
            infer.request = old
    return selection


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--source-cache", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--selection", required=True)
    a = p.parse_args()
    run(json.loads(Path(a.inputs).read_text()), a.source_cache, a.cache, a.selection)
