"""Twelve new claims on already-seen sources, not new documents or institutions."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from infer import ROOT


def build(cache):
    cache = Path(cache)
    previous = {
        r["id"]: r
        for f in ["official_inputs.json", "replication_inputs.json"]
        for r in json.loads((cache / f).read_text())
    }
    rows = []

    def add(original, claim, label, support):
        row = deepcopy(previous[original])
        row["id"] = "followup-" + original
        row["split"] = "followup_same_documents"
        row["claim"] = claim
        row["reference"] = dict(decision=label, evidence=support)
        rows.append(row)

    add(
        "jeju-old-sales",
        "최초 공고의 매출 기준만 묻는다. 사업을 2년 운영했고 전년도 매출은 1300만원인 가상 신청자는 이 매출 기준을 충족한다.",
        "supported",
        ["O020"],
    )
    add(
        "jeju-grandfather-old",
        "9월 10일에 최초 공고에 따라 이미 신청했다. 사업 운영 2년, 전년도 매출 1400만원, 최근 3개월 평균 월 80만원인 가상 신청자는 적용되는 매출 기준을 충족한다.",
        "supported",
        ["N004", "N006", "O020"],
    )
    add(
        "jeju-new-sales-both-low",
        "수정 공고의 매출 기준만 묻는다. 사업 운영 2년, 전년도 매출 1100만원, 최근 3개월 평균 월 95만원인 가상 신청자는 그 매출 기준을 충족한다.",
        "contradicted",
        ["N025"],
    )
    add(
        "jeju-new-birth",
        "실제 접수 가능 여부가 아니라 수정 공고에 인쇄된 2025년 출산일 범위만 묻는다. 12월 1일은 그 명시 범위 안이다.",
        "contradicted",
        ["N023"],
    )
    add(
        "jeju-possible-deadline",
        "접수기간 문구와 예산에 따른 변경 가능성만 제시되었다. 이후 실제 접수 운영 기록은 없다. 이 사업은 실제로 11월 28일에 접수를 끝냈다.",
        "not_established",
        ["N010", "N011"],
    )
    add(
        "jeju-actual-payout",
        "가상 신청자의 지원대상 확정이나 실제 지급 정보가 제시되지 않았다. 이 사람에게 실제 지급된 금액은 30만원이다.",
        "not_established",
        ["N015", "N031", "N032"],
    )
    add(
        "incheon-parent-residence",
        "인천 공공간호사 장학생의 명시된 1년 거주 요건만 묻는다. 신청일 현재 본인은 다른 지역에 거주하고 부모는 인천에 연속으로 정확히 1년 거주했다. 이 가상 신청자는 이 요건을 충족한다.",
        "supported",
        ["I05"],
    )
    add(
        "incheon-grade-boundary",
        "인천 공공간호사 장학생의 성적 요건만 묻는다. 전 학년 평점 평균이 3.1인 가상 신청자는 그 성적 요건을 충족한다.",
        "supported",
        ["I03"],
    )
    add(
        "incheon-split-payment",
        "인천 공공간호사 장학생 중 선발된 거주지 미충족자의 지급 절차만 묻는다. 전입신고 조건을 확인하기 전에 처음 지급하는 금액은 1000만원이다.",
        "contradicted",
        ["I12"],
    )
    add(
        "incheon-application-final",
        "인천 공공간호사 장학생의 명시된 신청서 접수기간만 묻는다. 2025년 4월 30일 오후 6시는 접수기간 안이다.",
        "contradicted",
        ["I02"],
    )
    add(
        "incheon-exception-not-selection",
        "인천 공공간호사 장학생의 거주 기간 미충족자가 확약서를 제출했다는 사실만 제시되었다. 이 사람의 최종 면접 결과와 선발은 확정되었다.",
        "not_established",
        ["I06", "I10"],
    )
    add(
        "incheon-actual-payment",
        "인천 공공간호사 장학생의 가상 신청자가 최종 선발되었는지, 지급받았는지 제시되지 않았다. 이 사람에게 실제 지급된 금액은 700만원이다.",
        "not_established",
        ["I11", "I12"],
    )
    (cache / "followup_inputs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    refs = [
        {k: v for k, v in row.items() if k != "sentences"}
        | {"sentence_ids": list(row["sentences"])}
        for row in rows
    ]
    return rows, refs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    _, refs = build(a.cache)
    (ROOT / "followup_reference.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2) + "\n"
    )
