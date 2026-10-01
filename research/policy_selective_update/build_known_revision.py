"""Development on the previously observed Jeju revision; not a held-out source."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from core import sha
from infer import ROOT


def build(cache):
    cache = Path(cache)
    docs = {}
    for phase, source in [("before", "jeju_before"), ("after", "jeju_after")]:
        docs[phase] = {
            r["id"]: r["text"]
            for r in json.loads((cache / (source + "-paragraphs.json")).read_text())
        }
    selected = {"before": ["P020", "P021"], "after": ["P004", "P006", "P025", "P026"]}
    bundle = {"id": "jeju-known", "kind": "previously_observed_source"}
    locators = {}
    for phase, prefix in [("before", "O"), ("after", "N")]:
        bundle[phase] = {prefix + p[1:]: docs[phase][p] for p in selected[phase]}
        locators[phase] = {
            prefix
            + p[1:]: {
                "document": "jeju_" + phase,
                "paragraph": p,
                "text_sha256": sha(docs[phase][p].encode()),
            }
            for p in selected[phase]
        }
    C, S, U = "contradicted", "supported", "not_established"
    rows = [
        ("기존", 900, 110, C, C, "keep", ["O020", "N006"]),
        ("신규", 900, 110, C, S, "recheck", ["N025", "N006"]),
        ("신규", 1300, 80, S, S, "recheck", ["N025", "N006"]),
        ("신규", 900, 80, C, C, "recheck", ["N025", "N006"]),
        ("기존", 1300, 80, S, S, "keep", ["O020", "N006"]),
        ("미상", 900, 110, C, U, "uncertain", ["O020", "N025", "N006"]),
    ]
    cases = []
    for i, (status, annual, monthly, before, after, route, ids) in enumerate(rows, 1):
        description = {
            "기존": "9월 10일 최초 공고에 따라 이미 신청했다.",
            "신규": "9월 20일 새로 신청할 예정이다.",
            "미상": "최초 공고에 이미 접수했는지 수정 공고 이후 신규 접수할지 알려지지 않았다.",
        }[status]
        cases.append(
            {
                "id": f"jeju-known-{i}",
                "bundle": bundle["id"],
                "claim": f"제주 소상공인 출산급여의 적용 매출 기준만 묻는다. 2년간 운영한 가상 신청자는 전년도 매출 {annual}만원, 최근 3개월 월평균 매출 {monthly}만원이다. {description} 이 사람은 적용되는 매출 기준을 충족한다.",
                "reference": {
                    "before": before,
                    "after": after,
                    "route": route,
                    "before_evidence": ["O020"],
                    "after_evidence": ids,
                },
            }
        )
    return {"bundles": [bundle], "cases": cases}, {
        "bundles": [
            {"id": bundle["id"], "kind": bundle["kind"], "source_locators": locators}
        ],
        "cases": cases,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    data, reference = build(a.cache)
    Path(a.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "known_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n"
    )
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Previously observed Jeju development, not new source evaluation",
        "private_input_sha256": sha(data),
        "artifacts": {
            n: sha((ROOT / n).read_bytes())
            for n in ["known_reference.json", "build_known_revision.py"]
        },
    }
    (ROOT / "known_freeze.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
