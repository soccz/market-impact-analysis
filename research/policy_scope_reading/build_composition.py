"""48 profile queries in six authored worlds, fixed before test predictions."""

import itertools
import json
from pathlib import Path

from logic import label, oracle

ROOT = Path(__file__).resolve().parent


def build():
    rows = []

    def add(family, sentences, claim, facts, evidence):
        n = sum(r["family"] == family for r in rows) + 1
        rows.append(
            dict(
                id=f"compose-{family}-{n}",
                split="composition",
                family=family,
                source_kind="authored_composition",
                sentences=sentences,
                claim=claim,
                oracle_facts=facts,
                reference=dict(
                    decision=label(oracle(family, facts)), evidence=evidence
                ),
            )
        )

    for old, first, income in itertools.product(
        [True, False], [True, False], [90, 125]
    ):
        sentences = {
            "S1": "가상 해솔사업의 10월 1일 전 접수 건은 종전 기준을, 10월 1일 이후 접수 건은 새 기준을 따른다.",
            "S2": "종전 소득 상한은 수혜 이력과 관계없이 100%이다.",
            "S3": "새 기준은 첫 신청자에게 140%, 재신청자에게 110%의 소득 상한을 적용한다.",
        }
        claim = f"9월 20일" if old else "10월 2일"
        claim += f"에 접수한 {'첫 신청자' if first else '재신청자'}의 소득이 {income}%이다. 이 사람은 적용되는 소득 상한을 충족한다."
        add(
            "version_scope",
            sentences,
            claim,
            dict(old=old, first=first, income=income),
            ["S1", "S2" if old else "S3"],
        )
    for education_only, credits, residence in itertools.product(
        [True, False], [6, 10], [4, 8]
    ):
        sentences = {
            "S1": "가상 푸른사업은 생계·의료·주거·교육급여 수급자를 원칙적으로 제외한다.",
            "S2": "예외로 교육급여만 받는 사람은 이수학점이 8학점 이상이면 이 급여 제외에 해당하지 않는다.",
            "S3": "모든 신청자는 지역 거주 기간이 6개월 이상이어야 한다.",
        }
        benefits = "교육급여만" if education_only else "생계급여와 교육급여를 함께"
        claim = f"{benefits} 받고 {credits}학점을 이수했으며 {residence}개월 거주한 가상 신청자는 제시된 급여·학점·거주 조건을 모두 충족한다."
        # A sufficient proof may be any failed conjunct. Keep full explanation as a
        # preferred set and explicitly admit the smaller sufficient alternatives.
        add(
            "only_exception",
            sentences,
            claim,
            dict(education_only=education_only, credits=credits, residence=residence),
            ["S1", "S2", "S3"],
        )
        sets = []
        if not education_only:
            sets.append(["S1", "S2"])
        if credits < 8:
            sets.append(["S1", "S2"])
        if residence < 6:
            sets.append(["S3"])
        rows[-1]["reference"]["support_sets"] = sets or [["S1", "S2", "S3"]]
    for first, annual, monthly in itertools.product(
        [True, False], [900, 1300], [90, 110]
    ):
        sentences = {
            "S1": "가상 마루사업의 신규 신청자는 연 매출 1200만원 이상 또는 최근 월평균 매출 100만원 이상이면 매출 요건을 충족한다.",
            "S2": "기존 수혜자가 다시 신청할 때에는 두 매출 기준을 모두 충족해야 한다.",
        }
        claim = f"{'신규 신청자' if first else '기존 수혜 재신청자'}의 연 매출은 {annual}만원, 최근 월평균 매출은 {monthly}만원이다. 제시된 매출 요건을 충족한다."
        add(
            "threshold_or",
            sentences,
            claim,
            dict(first=first, annual=annual, monthly=monthly),
            ["S1"] if first else ["S1", "S2"],
        )
    state_sentences = {
        "confirmed": "가상 새봄사업의 확정된 1인당 정액 지원금은 50만원이다.",
        "proposed": "가상 새봄사업의 정액 지원금을 50만원으로 정하는 방안을 검토 중이며 확정 금액은 아직 발표하지 않았다.",
        "denied": "가상 새봄사업의 확정 정액 지원금이 50만원이라는 발표는 사실이 아니라고 기관이 밝혔다. 다른 금액은 밝히지 않았다.",
        "missing": "가상 새봄사업의 정액 지원금은 별표에 정한다. 이 입력에는 별표가 제공되지 않았다.",
    }
    for state, amount in itertools.product(state_sentences, [50, 70]):
        add(
            "modality",
            {
                "S1": state_sentences[state],
                "S2": "별개의 가상 별빛사업은 정액으로 70만원을 지원한다.",
            },
            f"가상 새봄사업의 확정된 1인당 정액 지원금은 {amount}만원이다.",
            dict(state=state, amount=amount),
            ["S1"],
        )
    for target, business, elapsed in itertools.product(
        ["business", "payment"], [4, 8], [2, 4]
    ):
        sentences = {
            "S1": "가상 돌봄사업에서 출산일 기준 사업 운영 기간은 6개월 이상이어야 한다.",
            "S2": "지급일 기준 출산 후 3개월을 초과하면 일괄 지급의 경과 기간 조건을 충족한다.",
        }
        ending = (
            "출산일 기준 사업 운영 기간 조건"
            if target == "business"
            else "지급일 기준 일괄 지급의 경과 기간 조건"
        )
        claim = f"가상 신청자는 출산일까지 사업을 {business}개월 운영했고, 지급일에는 출산 후 {elapsed}개월이 지났다. 이 사람은 {ending}을 충족한다."
        add(
            "anchor_dates",
            sentences,
            claim,
            dict(target=target, business_months=business, elapsed_months=elapsed),
            ["S1" if target == "business" else "S2"],
        )
    for actual, amount, wording in itertools.product(
        [True, False], [600000, 1800000], [0, 1]
    ):
        sentences = {
            "S1": "가상 온기사업의 지자체 지원은 월 최대 20만원씩 3개월이며, 총 상한은 60만원이다.",
            "S2": "별도 국비 지원의 총 상한은 120만원이다.",
            "S3": "실제 지급은 대상자 확정과 인정 비용에 따른다. 이 가상 신청자의 지급 결정은 제공되지 않았다.",
        }
        amount_text = f"{amount}원" if wording == 0 else f"{amount//10000}만원"
        claim = f"가상 온기사업에서 {'이 신청자의 실제 지자체 지급액' if actual else '지자체 지원 총 상한'}은 {amount_text}이다."
        add(
            "role_units",
            sentences,
            claim,
            dict(actual=actual, amount_won=amount, wording=wording),
            ["S1", "S3"] if actual else ["S1"],
        )
    assert len(rows) == 48
    return rows


if __name__ == "__main__":
    (ROOT / "composition.json").write_text(
        json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    )
