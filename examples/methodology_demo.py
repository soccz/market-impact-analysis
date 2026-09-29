"""Re-run a public relevance function on invented inputs; standard library only.

This 2026-09 documentation example is separate from the historical experiments.
It does not load articles, models, prices, or any client deliverable.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "v2/code/ptei_utils.py"
POLICY = {
    "kw1": "제조지원안",
    "kw2": "설비 투자 공제",
    "alias": "가상 설비지원 정책",
    "top_stocks": "한빛제조",
    "not_words": "",
}
CASES = [
    {
        "id": "policy",
        "label": "정책을 언급하면",
        "title": "제조지원안 발표, 신규 설비 투자 지원 확대",
        "content": "정부가 생산 설비 투자의 부담을 낮추는 가상 지원안을 발표했다.",
        "expected_level": "L1_primary_exact",
        "expected_score": 1.0,
        "interpretation": "정책 이름이 입력 규칙과 일치합니다. 다음으로 이 기사가 어떤 방향의 근거인지 읽어야 합니다.",
    },
    {
        "id": "reversal",
        "label": "정책을 철회해도",
        "title": "제조지원안 철회, 신규 설비 투자 지원 취소",
        "content": "정부가 당초 계획했던 가상 지원안을 철회했다.",
        "expected_level": "L1_primary_exact",
        "expected_score": 1.0,
        "interpretation": "철회 기사도 같은 정책에 관한 기사입니다. 관련성이 같아도 지지·반박의 방향은 달라질 수 있습니다.",
    },
    {
        "id": "company",
        "label": "기업 이름만 있으면",
        "title": "한빛제조, 신제품 생산 계획 발표",
        "content": "회사가 새로운 제품의 출시 일정을 소개했다.",
        "expected_level": "NO_MATCH",
        "expected_score": 0.0,
        "interpretation": "기업 이름은 발견했지만 정책 근거는 찾지 못했습니다. 종목명만으로 관련 정책 기사라고 승인하지 않는 선택입니다.",
    },
    {
        "id": "paraphrase",
        "label": "표현을 바꾸면",
        "title": "생산시설을 새로 짓는 기업의 세금 부담을 낮춘다",
        "content": "정부가 제조기업의 생산시설 신설에 세제 혜택을 제공한다.",
        "expected_level": "NO_MATCH",
        "expected_score": 0.0,
        "interpretation": "같은 취지를 설명하도록 만든 가상 문장이지만 등록된 정책 표현은 없습니다. 규칙으로 근거를 추적하기 쉬워진 만큼 표현 변형을 놓칠 수 있습니다.",
    },
]


def build_report():
    spec = importlib.util.spec_from_file_location("historical_ptei_utils", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for case in CASES:
        score, detail = module.relevance_v3_1(case["title"], case["content"], POLICY)
        cases.append({**case, "score": score, "detail": detail})
    return {
        "synthetic_only": True,
        "source": "v2/code/ptei_utils.py",
        "function": "relevance_v3_1",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "policy": POLICY,
        "cases": cases,
        "scope": "Relevance rules only; no stance model or market evaluation.",
    }


def verify(report):
    for case in report["cases"]:
        if (case["score"], case["detail"]["level"]) != (
            case["expected_score"],
            case["expected_level"],
        ):
            raise AssertionError(f"Changed behavior: {case['id']}")
    company = next(case for case in report["cases"] if case["id"] == "company")
    if company["detail"]["stock_hits"] != ["한빛제조"]:
        raise AssertionError("Company example must actually match the company name")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check the documented behavior"
    )
    args = parser.parse_args()
    report = build_report()
    if args.check:
        verify(report)
        print(
            "PASS: 4 synthetic cases, historical relevance function, no external data"
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
