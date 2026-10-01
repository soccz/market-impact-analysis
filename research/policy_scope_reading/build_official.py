"""Source-backed claim probes; source text is rebuilt into a private cache."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from prepare_sources import ROOT, prepare, sha


def build(cache):
    docs = prepare(cache)
    rows = []

    def add(name, family, ids, claim, label, support=None):
        sentences = {
            key: docs["jeju_before" if key.startswith("O") else "jeju_after"][
                "P" + key[1:]
            ]
            for key in ids
        }
        refs = support or ids
        rows.append(
            dict(
                id="jeju-" + name,
                split="official",
                family=family,
                source_kind="official_excerpt",
                sentences=sentences,
                claim=claim,
                reference=dict(decision=label, evidence=refs),
                source_locators={
                    key: dict(
                        document="jeju_before" if key.startswith("O") else "jeju_after",
                        paragraph="P" + key[1:],
                        sha256=sha(value.encode()),
                    )
                    for key, value in sentences.items()
                },
            )
        )

    add(
        "old-sales",
        "recipient_and_alternative",
        ["O020", "O021"],
        "최초 공고의 매출 기준만 묻는다. 2년간 운영했고 전년도 매출 900만원, 최근 3개월 평균 월 매출 110만원인 가상 신청자는 그 매출 기준을 충족한다.",
        "contradicted",
        ["O020"],
    )
    add(
        "new-sales",
        "recipient_and_alternative",
        ["N025", "N026"],
        "수정 공고의 매출 기준만 묻는다. 2년간 운영했고 전년도 매출 900만원, 최근 3개월 평균 월 매출 110만원인 가상 신청자는 그 매출 기준을 충족한다.",
        "supported",
        ["N025"],
    )
    add(
        "new-sales-both-low",
        "recipient_and_alternative",
        ["N025", "N026"],
        "수정 공고의 매출 기준만 묻는다. 2년간 운영했고 전년도 매출 900만원, 최근 3개월 평균 월 매출 90만원인 가상 신청자는 그 매출 기준을 충족한다.",
        "contradicted",
        ["N025"],
    )
    add(
        "new-sales-annual",
        "recipient_and_alternative",
        ["N025", "N026"],
        "수정 공고의 매출 기준만 묻는다. 2년간 운영했고 전년도 매출 1300만원, 최근 3개월 평균 월 매출 80만원인 가상 신청자는 그 매출 기준을 충족한다.",
        "supported",
        ["N025"],
    )
    add(
        "grandfather-old",
        "version_applicability",
        ["N004", "N006", "O020", "O021", "N025"],
        "9월 10일에 최초 공고에 따라 이미 신청했다. 사업 운영 2년, 전년도 매출 900만원, 최근 3개월 평균 월 110만원인 가상 신청자는 적용되는 매출 기준을 충족한다.",
        "contradicted",
        ["N004", "N006", "O020"],
    )
    add(
        "grandfather-new",
        "version_applicability",
        ["N004", "N006", "O020", "O021", "N025"],
        "9월 20일에 신청했다. 사업 운영 2년, 전년도 매출 900만원, 최근 3개월 평균 월 110만원인 가상 신청자는 적용되는 매출 기준을 충족한다.",
        "supported",
        ["N004", "N006", "N025"],
    )
    add(
        "old-birth",
        "separate_date_anchors",
        ["O015", "O041"],
        "실제 접수 가능 여부가 아니라 최초 공고에 인쇄된 2025년 출산일 범위만 묻는다. 11월 29일은 그 명시 범위 안이다.",
        "contradicted",
        ["O015"],
    )
    add(
        "new-birth",
        "separate_date_anchors",
        ["N023", "N010"],
        "실제 접수 가능 여부가 아니라 수정 공고에 인쇄된 2025년 출산일 범위만 묻는다. 11월 29일은 그 명시 범위 안이다.",
        "supported",
        ["N023"],
    )
    add(
        "new-application",
        "separate_date_anchors",
        ["N023", "N010"],
        "출산일 범위가 아니라 수정 공고에 인쇄된 접수기간만 묻는다. 2025년 11월 29일은 그 명시 접수기간 안이다.",
        "contradicted",
        ["N010"],
    )
    add(
        "possible-deadline",
        "modality",
        ["N010", "N011"],
        "예산 소진에 따른 변경 가능성까지 고려하더라도 접수 종료일은 반드시 11월 28일로 유지된다고 확정할 수 있다.",
        "not_established",
        ["N011"],
    )
    add(
        "local-cap",
        "amount_role",
        ["N015", "N016"],
        "국비 출산급여와 구분한 이 제주 추가지원의 총 상한은 240만원이다.",
        "contradicted",
        ["N015", "N016"],
    )
    add(
        "actual-payout",
        "modality",
        ["N015", "N031", "N032"],
        "가상 신청자의 지원대상 확정·지급 결정 정보는 제시되지 않았다. 이 사람의 실제 지급액은 정확히 90만원이다.",
        "not_established",
        ["N015", "N031", "N032"],
    )
    add(
        "payment-anchor",
        "separate_date_anchors",
        ["N028", "N046"],
        "이 두 지급 문구는 모두 신청일을 기준으로 출산 후 경과 기간을 계산한다.",
        "contradicted",
        ["N028", "N046"],
    )
    add(
        "new-application-elapsed",
        "separate_date_anchors",
        ["N028", "N046"],
        "유의사항에 적힌 신청 시점의 일괄지급 기준만 묻는다. 출산 후 2.5개월에 신청한 가상 신청자는 그 경과 기간 기준을 충족한다.",
        "supported",
        ["N046"],
    )
    add(
        "old-application-elapsed",
        "separate_date_anchors",
        ["O040"],
        "최초 공고 유의사항의 신청 시점 일괄지급 기준만 묻는다. 출산 후 2.5개월에 신청한 가상 신청자는 그 경과 기간 기준을 충족한다.",
        "contradicted",
        ["O040"],
    )
    add(
        "new-payment-elapsed",
        "separate_date_anchors",
        ["N028", "N046"],
        "지급방식에 적힌 지급일 기준만 묻는다. 지급일에 출산 후 3.5개월이 지난 가상 신청자는 그 경과 기간 기준을 충족한다.",
        "supported",
        ["N028"],
    )
    add(
        "old-short-business",
        "recipient_and_alternative",
        ["O020", "O021"],
        "최초 공고의 매출 기준만 묻는다. 사업 운영 기간이 8개월이고 월 매출 110만원을 증빙한 가상 신청자는 6~11개월 사업자의 명시 월 매출 기준을 충족한다.",
        "supported",
        ["O021"],
    )
    add(
        "duplicate-program",
        "other_program",
        ["O044", "O045"],
        "국비 고용보험 미적용자 출산급여 수급자는 제주 1인 소상공인 출산급여 지원 사업과 중복신청할 수 없다고 이 근거는 명시한다.",
        "contradicted",
        ["O044", "O045"],
    )
    base = list(rows)
    for row in base:
        for variant in ["distractor", "removed"]:
            item = json.loads(json.dumps(row))
            item["id"] += "-" + variant
            item["variant"] = variant
            item["base_id"] = row["id"]
            # Newly authored distractors are explicitly different programmes, never
            # represented as quotations from the real issuer.
            extra = {
                "X1": "별개의 가상 누리사업은 전년도 매출 1500만원 이상을 요구하며, 해당 사업의 총 지원 상한은 240만원이다.",
                "X2": "별개의 가상 누리사업 접수 마감은 12월 20일이며 첫 신청자만 지원한다.",
            }
            if variant == "removed":
                item["sentences"] = {
                    "X0": "실험용 근거 가림: 이 입력에는 주장에서 묻는 실제 사업의 해당 조항이 제공되지 않았다.",
                    **extra,
                }
                item["source_locators"] = {}
                item["source_kind"] = "source_withheld_control"
                item["reference"] = dict(decision="not_established", evidence=["X0"])
            else:
                item["sentences"] = {**extra, **item["sentences"]}
                item["source_kind"] = "official_excerpt_with_authored_distractors"
            rows.append(item)
    return rows


def write(cache):
    rows = build(cache)
    private = Path(cache) / "official_inputs.json"
    private.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    public = []
    for row in rows:
        item = {k: v for k, v in row.items() if k != "sentences"}
        item["input_sha256"] = sha(
            json.dumps(
                {"sentences": row["sentences"], "claim": row["claim"]},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        )
        item["sentence_ids"] = list(row["sentences"])
        item["sentence_hashes"] = {
            k: sha(v.encode()) for k, v in row["sentences"].items()
        }
        public.append(item)
    (ROOT / "official_reference.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n"
    )
    (ROOT / "official_annotation_record.json").write_text(
        json.dumps(
            dict(
                created_utc=datetime.now(timezone.utc).isoformat(),
                status="AI-assisted provisional reference before official predictions; model prompts frozen before full source reading, but search excerpts seen earlier. Manually selected source paragraphs: not automatic whole-document retrieval.",
                source_documents=2,
                revision_pairs=1,
                base_claims=18,
                authored_perturbations=36,
                independent_human_reviews=0,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        "Official: 18 source-backed claims + 18 distractor + 18 evidence-withheld controls; source text private."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    write(a.cache)
