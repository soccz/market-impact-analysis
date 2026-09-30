"""Short primary-source excerpts. Selection and labels are provisional, not human gold."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from build_data import amounts, write

ROOT = Path(__file__).resolve().parent
CASES = [
    (
        "water-budget",
        "148932342",
        "농림축산식품부",
        "수리시설개보수 및 배수개선사업 예산",
        "2025년 수리시설개보수 및 배수개선사업 예산은 재정 당국과 협의 중으로 아직 확정된 바 없습니다.",
        "unspecified",
        "under_review",
        [],
        [],
        ["재정 당국과 협의 중으로 아직 확정된 바 없습니다."],
    ),
    (
        "voucher-budget",
        "148930563",
        "농림축산식품부",
        "농식품 바우처 예산",
        "2025년 농식품 바우처 예산은 재정 당국과 협의 중으로 아직 확정된 바 없습니다.",
        "unspecified",
        "under_review",
        [],
        [],
        ["재정 당국과 협의 중으로 아직 확정된 바 없습니다."],
    ),
    (
        "cbam-expansion",
        "148932239",
        "산업통상자원부",
        "CBAM 적용 품목 확대",
        "EU CBAM이 2030년에 플라스틱, 자동차, 반도체, 배터리 등 전 품목으로 확장될 예정이라는 보도는 사실과 다릅니다.",
        "policy_change",
        "denied",
        [],
        ["전 품목으로 확장될 예정"],
        ["사실과 다릅니다."],
    ),
    (
        "nuclear-structure",
        "148963183",
        "산업통상부",
        "원전 수출체계 효율화 방안",
        "원전 수출체계 효율화 방안은 결정된 바 없습니다",
        "policy_change",
        "under_review",
        [],
        ["수출체계 효율화"],
        ["결정된 바 없습니다"],
    ),
    (
        "health-living",
        "148931883",
        "보건복지부",
        "건강생활유지비",
        "본인부담금 지원을 위한 건강생활유지비도 월 6000원에서 1만 2000원으로 2배 인상한다.",
        "policy_change",
        "asserted",
        ["before", "after"],
        ["2배 인상한다."],
        [],
    ),
    (
        "coupon-funding",
        "148946134",
        "관계부처 합동",
        "필수 예산 삭감을 통한 소비쿠폰 재원 마련",
        "필수 예산을 삭감해 민생회복 소비쿠폰의 재원이 마련되었다는 주장은 전혀 사실이 아닙니다.",
        "policy_change",
        "denied",
        [],
        ["필수 예산을 삭감해"],
        ["전혀 사실이 아닙니다."],
    ),
    (
        "officer-cut",
        "148934797",
        "기획재정부·국방부",
        "초급간부 처우개선 예산 대폭 삭감",
        "내년 초급간부 처우개선 예산 대폭 삭감은 사실이 아닙니다.",
        "policy_change",
        "denied",
        [],
        ["예산 대폭 삭감"],
        ["사실이 아닙니다."],
    ),
]


def evidence(text, relations, states):
    return [
        dict(start=text.index(t), end=text.index(t) + len(t), text=t, kind=kind)
        for kind, terms in [("relation", relations), ("state", states)]
        for t in terms
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    assert not (ROOT / "freeze.json").exists()
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []
    old = [
        json.loads(s)
        for s in (ROOT.parent / "policy_revisions/data/official_cases.jsonl")
        .read_text()
        .splitlines()
    ]
    queries = [
        "공사비",
        "자립수당",
        "원주사랑상품권 발행규모",
        "영아수당",
        "산불재난특수진화대 증원 및 예산 규모",
    ]
    for x, q in zip(old, queries):
        row = dict(
            x,
            query=q,
            state=(
                "planned"
                if x["id"] == "infant-plan"
                else "under_review" if x["id"] == "forest-unconfirmed" else "asserted"
            ),
            split="official",
            condition="official",
            composition="not_applicable",
            known_source=True,
            pair_id=x["id"],
        )
        row["relation"] = (
            "policy_change"
            if x["id"] == "infant-plan"
            else "unspecified" if x["relation"] == "undetermined" else x["relation"]
        )
        row["spans"] = amounts(row["text"])
        for i, sp in enumerate(row["spans"]):
            sp["role"] = ["before", "after"][i]
        terms = {
            "incheon-error": (["산출 오류에 따른 공사비 정정"], []),
            "youth-increase": (["인상하고"], []),
            "wonju-increase": (["대폭 늘렸다."], ["늘렸다."]),
            "infant-plan": (["단계적으로 인상해"], ["지급할 계획이다."]),
            "forest-unconfirmed": ([], ["확정된 바 없습니다."]),
        }[x["id"]]
        row["evidence"] = evidence(row["text"], *terms)
        row["state_ambiguity"] = (
            ["asserted", "planned"] if x["id"] == "youth-increase" else []
        )
        row[
            "note"
        ] += " v3 labels and prospective roles newly assigned; v2 immutable. Non-finalized is not a denial."
        rows.append(row)
    for key, news, publisher, query, quote, rel, state, roles, reasons, status in CASES:
        url = (
            "https://www.korea.kr/"
            + (
                "news/policyNewsView.do?newsId="
                if news in ["148931883", "148934797"]
                else "briefing/actuallyView.do?newsId="
            )
            + news
        )
        cache = args.cache / (news + ".txt")
        if cache.exists():
            text = cache.read_text()
        else:
            response = requests.get(url, timeout=40)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for t in soup(["script", "style"]):
                t.decompose()
            text = " ".join(soup.get_text(" ").split())
            cache.write_text(text)
        assert quote in text, key
        assert len(quote.split()) <= 25, (key, len(quote.split()))
        assert "공공누리 제1유형" in text or "공공누리 출처표시" in text, key
        dates = re.findall(r"20\d\d\.\d\d\.\d\d", text)
        spans = amounts(quote)
        assert len(spans) == len(roles), (key, spans)
        for sp, role in zip(spans, roles):
            sp["role"] = role
        start = text.index(quote)
        rows.append(
            dict(
                id=key,
                pair_id=key,
                query=query,
                text=quote,
                relation=rel,
                state=state,
                spans=spans,
                evidence=evidence(quote, reasons, status),
                url=url,
                publisher=publisher,
                date=dates[0].replace(".", "-"),
                source_normalized_sha256=hashlib.sha256(text.encode()).hexdigest(),
                excerpt_start=start,
                excerpt_end=start + len(quote),
                retrieved_utc=datetime.now(timezone.utc).isoformat(),
                reuse="Short unmodified attributed text; KOGL type 1 notice observed. Full source cache not redistributed.",
                label_status="AI provisional; independent human adjudication pending",
                split="official",
                condition="official",
                composition="not_applicable",
                known_source=False,
                state_ambiguity=(
                    ["asserted", "planned"] if key == "health-living" else []
                ),
                note="Selected short passage, manually specified query. Authority statement is not independent verification of reality. No fabricated before/after document pair.",
            )
        )
    write("official", rows)
    (ROOT / "data/source_selection.json").write_text(
        json.dumps(
            dict(
                included=len(rows),
                known=5,
                newly_inspected=7,
                selection="Purposive scope and attribution examples; neither random nor comprehensive crawl.",
                excluded=[
                    dict(
                        url="https://www.korea.kr/news/policyNewsView.do?newsId=148938402",
                        reason="KOGL type 4 notice: not added to this reusable experiment bundle.",
                    )
                ],
                missing=[
                    "Independent annotators",
                    "Paired document version collection",
                    "Balanced official classes: no restatement or hypothetical example",
                ],
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("official cases", len(rows))


if __name__ == "__main__":
    main()
