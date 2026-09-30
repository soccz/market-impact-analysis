"""Small attributed applicability audit; labels are provisional AI judgments.

Not a random sample, not human gold, and not part of the primary synthetic test.
Four sources were already inspected in v1; Incheon is newly inspected for v2.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup
import requests

from model import amounts

ROOT = Path(__file__).resolve().parent
CASES = [
    (
        "incheon-error",
        "https://www.incheon.go.kr/jonggeon/JO020101/3008010",
        "인천광역시 종합건설본부",
        "2025-04-21",
        "공사비 산출 오류에 따른 공사비 정정",
        "169,206천원",
        "correction",
        ["previous", "current"],
        "Newly inspected source; explicit calculation error. Before/after report values, not a claim that actual spending changed.",
    ),
    (
        "youth-increase",
        "https://www.korea.kr/news/policyNewsView.do?newsId=148920503",
        "보건복지부·정책브리핑",
        "2023-09-19",
        "자립준비 청년에게 지원되는 자립수당은",
        "민간 협력을 강화한다.",
        "policy_change",
        ["previous", "current"],
        "Known v1 source. Announced benefit increase; passage-level relation, not proof of later implementation.",
    ),
    (
        "wonju-increase",
        "https://www.wonju.go.kr/media/selectBbsNttView.do?bbsNo=145&key=3450&nttNo=416729",
        "원주시",
        "2023",
        "올해 원주사랑상품권 발행규모를",
        "대폭 늘렸다.",
        "policy_change",
        ["previous", "current"],
        "Known v1 source. Issuance level increased, not reported error or increment value.",
    ),
    (
        "infant-plan",
        "https://www.korea.kr/news/policyNewsView.do?newsId=148881122",
        "저출산고령사회위원회·정책브리핑",
        "2020-12-15",
        "특히 2022년도 출생아부터",
        "매월 지급할 계획이다.",
        "undetermined",
        ["other", "other"],
        "Known v1 source. A multi-year future plan; no asserted current/previous implemented amount under this study's narrow schema. Alternative plan-aware schema is needed.",
    ),
    (
        "forest-unconfirmed",
        "https://www.korea.kr/briefing/actuallyView.do?newsId=148970350",
        "기획예산처·정책브리핑",
        "2026-08-20",
        "산불재난특수진화대 증원 및 예산 규모는",
        "확정된 바 없습니다.",
        "undetermined",
        [],
        "Known v1 source; title-only evidence and no monetary values.",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    assert not (ROOT / "data/official_cases.jsonl").exists()
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for identifier, url, publisher, date, begin, end, relation, roles, note in CASES:
        response = requests.get(url, timeout=40)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        text = " ".join(soup.get_text(" ").split())
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        excerpt = text[start:stop]
        assert len(excerpt.split()) <= 25 and len(excerpt) < 220, excerpt
        spans = amounts(excerpt)
        assert len(spans) == len(roles), (identifier, spans)
        for span, role in zip(spans, roles):
            span["role"] = role
        (args.cache / f"{identifier}.txt").write_text(text)
        rows.append(
            {
                "id": identifier,
                "text": excerpt,
                "relation": relation,
                "spans": spans,
                "url": url,
                "publisher": publisher,
                "date": date,
                "note": note,
                "label_status": "AI provisional; independent human adjudication pending",
                "source_normalized_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "excerpt_start": start,
                "excerpt_end": stop,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "reuse": "Short attributed excerpt; KOGL attribution notice observed on source page. No full-page redistribution.",
            }
        )
        print(identifier, excerpt, flush=True)
    (ROOT / "data/official_cases.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )


if __name__ == "__main__":
    main()
