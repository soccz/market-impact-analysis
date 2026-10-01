"""A second institution, annotated after method freeze and before prediction."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import fitz
from infer import ROOT


def build(cache):
    cache = Path(cache)
    raw = (cache / "incheon-corrected.pdf").read_bytes()
    meta = json.loads((cache / "incheon-retrieval.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == meta["sha256"]
    pages = [
        re.sub(r"\s+", " ", p.get_text()).strip()
        for p in fitz.open(stream=raw, filetype="pdf")
    ]
    spans = {}

    def excerpt(key, page, start, end):
        text = pages[page - 1]
        a = text.index(start)
        b = text.index(end, a)
        spans[key] = dict(page=page, start=a, end=b, text=text[a:b].strip())

    excerpt("I01", 1, "○선발공고기간", "○신청서 접수 기간")
    excerpt("I02", 1, "○신청서 접수 기간", "○신청 접수는")
    excerpt("I03", 1, "- 전 학년 평점", "- 대한민국 국적")
    excerpt("I04", 1, "- 대한민국 국적", "- 신청일 기준")
    excerpt("I05", 1, "- 신청일 기준", "※ 거주 미충족자")
    excerpt("I06", 1, "※ 거주 미충족자", "- 대학 졸업 후")
    excerpt("I07", 1, "- 대학 졸업 후", "- 남성의 경우")
    excerpt("I08", 1, "- 남성의 경우", "- 인천광역시의료원 인사규정")
    excerpt("I09", 2, "○서류 합격자", "○면접심사")
    excerpt("I10", 2, "○면접심사", "5. 선발 일정")
    excerpt("I11", 3, "○선발된 장학생은", "○선발된 장학생 중")
    excerpt("I12", 3, "○선발된 장학생 중", "○장학금 수령 후")
    rows = []

    def add(name, family, ids, claim, label, support=None):
        rows.append(
            dict(
                id="incheon-" + name,
                family=family,
                split="replication",
                source_kind="official_excerpt",
                sentences={k: spans[k]["text"] for k in ids},
                claim=claim,
                reference=dict(decision=label, evidence=support or ids),
                source_locators={
                    k: {
                        **{x: spans[k][x] for x in ["page", "start", "end"]},
                        "sha256": hashlib.sha256(spans[k]["text"].encode()).hexdigest(),
                    }
                    for k in ids
                },
            )
        )

    add(
        "parent-residence",
        "recipient_or",
        ["I05", "I06"],
        "인천 공공간호사 장학생의 거주 요건만 묻는다. 신청일 현재 본인은 타 지역 거주지만 부모는 인천에 계속 2년 거주 중이다. 이 가상 신청자는 명시된 1년 거주 요건을 충족한다.",
        "supported",
        ["I05"],
    )
    add(
        "exception",
        "exception",
        ["I05", "I06"],
        "인천 공공간호사 장학생의 거주 요건만 묻는다. 본인과 부모 모두 인천 거주 1년을 채우지 못했다. 전입 및 2년 이상 거주 확약서를 제출해도 이 거주 요건 때문에 반드시 제외된다.",
        "contradicted",
    )
    add(
        "exception-not-selection",
        "modality",
        ["I05", "I06", "I10"],
        "인천 공공간호사 장학생 신청자 중 거주 기간 미충족자가 전입 및 2년 이상 거주 확약서를 냈다. 다른 조건과 심사 결과가 없는 상태에서 이 가상 신청자의 최종 선발은 확정되었다.",
        "not_established",
        ["I06", "I10"],
    )
    add(
        "grade-boundary",
        "inclusive_boundary",
        ["I03"],
        "인천 공공간호사 장학생의 성적 요건만 묻는다. 전 학년 평점 평균이 정확히 3.0인 가상 신청자는 이 성적 요건을 충족한다.",
        "supported",
    )
    add(
        "grade-low",
        "inclusive_boundary",
        ["I03"],
        "인천 공공간호사 장학생의 성적 요건만 묻는다. 전 학년 평점 평균이 2.9인 가상 신청자는 이 성적 요건을 충족한다.",
        "contradicted",
    )
    add(
        "application-notice",
        "date_anchor",
        ["I01", "I02"],
        "인천 공공간호사 장학생의 명시된 신청서 접수기간만 묻는다. 2025년 4월 10일은 접수기간 안이다.",
        "contradicted",
        ["I02"],
    )
    add(
        "application-final",
        "date_anchor",
        ["I01", "I02"],
        "인천 공공간호사 장학생의 명시된 신청서 접수기간만 묻는다. 2025년 4월 30일 오후 4시는 접수기간 안이다.",
        "supported",
        ["I02"],
    )
    add(
        "nationality-exception",
        "exception",
        ["I04"],
        "인천 공공간호사 장학생의 국적·신분 조건만 묻는다. 대한민국 국적을 소지했지만 영주권자인 가상 신청자는 이 조건을 충족한다.",
        "contradicted",
    )
    add(
        "split-payment",
        "payment_scope",
        ["I11", "I12"],
        "인천 공공간호사 장학생 중 선발된 거주지 미충족자의 명시된 지급 절차만 묻는다. 전입신고 조건을 확인하기 전에 처음 지급하는 금액은 700만원이다.",
        "supported",
        ["I12"],
    )
    add(
        "unconditional-payment",
        "payment_scope",
        ["I11", "I12"],
        "인천 공공간호사 장학생 중 선발된 거주지 미충족자는 전입신고 기한을 지키지 않아도 나머지 300만원을 조건 없이 지급받는다.",
        "contradicted",
        ["I12"],
    )
    add(
        "actual-payment",
        "modality",
        ["I11", "I12"],
        "인천 공공간호사 장학생의 가상 신청자에 관해 선발·전입·지급 결정은 제시되지 않았다. 이 사람에게 실제 지급된 금액은 정확히 1000만원이다.",
        "not_established",
    )
    add(
        "shortlist-final",
        "selection_stage",
        ["I09", "I10"],
        "인천 공공간호사 장학생 선발에서 서류 합격했다는 정보만 있다. 이 가상 신청자의 면접 결과와 최종 선발은 이미 확정되었다.",
        "not_established",
    )
    distractors = {
        "Z1": "다른 가상 사업인 한빛 돌봄 장학생은 본인이 한빛시에 3년 거주해야 하며 성적은 2.5 이상이다.",
        "Z2": "한빛 돌봄 장학생의 접수는 4월 1일부터이며 선발자는 전입 조건 없이 300만원을 받는다.",
    }
    variants = []
    for row in rows:
        d = deepcopy(row)
        d["id"] += "-distractor"
        d["variant"] = "distractor"
        d["sentences"] = {
            "Z1": distractors["Z1"],
            **d["sentences"],
            "Z2": distractors["Z2"],
        }
        variants.append(d)
        d = deepcopy(row)
        d["id"] += "-removed"
        d["variant"] = "removed"
        d["sentences"] = {
            "Z0": "실험에서 인천 공공간호사 장학생의 해당 근거는 가려졌다. 다음 문장은 별도 가상 사업의 정보이다.",
            **distractors,
        }
        d["reference"] = dict(decision="not_established", evidence=["Z0"])
        d["source_locators"] = {}
        variants.append(d)
    rows += variants
    (cache / "replication_inputs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    refs = [
        {k: v for k, v in r.items() if k != "sentences"}
        | {"sentence_ids": list(r["sentences"])}
        for r in rows
    ]
    return rows, refs, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    rows, refs, meta = build(a.cache)
    (ROOT / "replication_reference.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2) + "\n"
    )
    (ROOT / "replication_source.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    print(dict(cases=len(rows), base=12, real_documents=1, revision_pair=False))
