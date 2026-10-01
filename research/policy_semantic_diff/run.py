"""Reproduce the finite scenario audit with Python's standard library only."""

import argparse
from collections import Counter
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
from engine import compare

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    cases = json.loads((ROOT / "cases.json").read_text())
    grid = protocol["grid"]
    rows = []
    for i, (d, r, b) in enumerate(
        product(grid["deposit_won"], grid["rent_won"], grid["benefits"])
    ):
        profile = {"deposit_won": d, "rent_won": r, "benefits": b}
        result = compare(profile)
        rows.append(
            {
                "id": f"grid-{i:04d}",
                "profile": profile,
                "result": result,
                "drop_exception": compare(profile, ablation="drop_exception"),
                "keep_old_welfare": compare(profile, ablation="keep_old_welfare"),
            }
        )
    counts = dict(sorted(Counter(x["result"]["change"] for x in rows).items()))
    known = [x for x in rows if x["result"]["change"] != "unresolved"]
    controls = {}
    for name in ["drop_exception", "keep_old_welfare"]:
        mismatches = [x for x in known if x[name]["change"] != x["result"]["change"]]
        controls[name] = {
            "disagreements": len(mismatches),
            "determinate_reference_rows": len(known),
            "by_reference_change": dict(
                sorted(Counter(x["result"]["change"] for x in mismatches).items())
            ),
            "ids": [x["id"] for x in mismatches],
        }
    controls["cap_only"] = {
        "old_cap_won": 400000,
        "new_cap_won": 400000,
        "change_detected": False,
        "missed_direction_changes": counts.get("included", 0)
        + counts.get("excluded", 0),
    }
    public_cases = []
    for case in cases:
        row = dict(case)
        row["modes"] = {
            "complete": compare(case["profile"]),
            "hide_old": compare(
                case["profile"], hidden=["2022-housing", "2022-welfare"]
            ),
            "hide_new": compare(
                case["profile"], hidden=["2023-housing", "2023-welfare"]
            ),
        }
        public_cases.append(row)
    result = {
        "grid_rows": len(rows),
        "version_pairs": 1,
        "independent_human_reviews": 0,
        "counts": counts,
        "controls": controls,
        "input_sha256": {
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
            for n in [
                "engine.py",
                "run.py",
                "protocol.json",
                "sources.json",
                "cases.json",
                "test_engine.py",
            ]
        },
    }
    save(output / "summary.json", result)
    save(output / "scenarios.json", public_cases)
    with (output / "grid.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (output / "grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "deposit_won",
                "rent_won",
                "benefits",
                "old",
                "new",
                "change",
                "minimal_rule_changes",
                "without_exception",
                "without_welfare_change",
            ]
        )
        for x in rows:
            p, r = x["profile"], x["result"]
            writer.writerow(
                [
                    x["id"],
                    p["deposit_won"],
                    p["rent_won"],
                    "+".join(p["benefits"]) or "none",
                    r["2022"]["state"],
                    r["2023"]["state"],
                    r["change"],
                    json.dumps(r["minimal_rule_changes"]),
                    x["drop_exception"]["change"],
                    x["keep_old_welfare"]["change"],
                ]
            )
    report = f"""# 두 공고의 조건을 실행한 기록

2022-2542 / 2023-1362의 주거 조건과 특정 복지급여 제외만 비교한 탐색적 사례 연구입니다.
실제 문서 쌍 **1개**, 독립 사람 검수 **0명**. 조건은 AI 지원으로 수동 형식화했습니다.
{len(rows):,}행은 미리 명시한 금액 경계값 × 급여 유형의 가상 조합이며 독립 정책 사례 수가 아닙니다.

| 선택 조건의 변화 | 가상 조합 수 |
|---|---:|
"""
    labels = {
        "included": "미충족 → 충족",
        "excluded": "충족 → 미충족",
        "retained": "충족 유지",
        "outside": "미충족 유지",
        "unresolved": "해석 확인 필요",
    }
    for name, count in counts.items():
        report += f"| {labels[name]} | {count} |\n"
    report += f"""
상한 40만원만 비교하면 변화 없음으로 처리하지만, 선택 조건에서는 {controls['cap_only']['missed_direction_changes']}조합의 충족 여부가 바뀝니다.
이 수는 수혜자 증가·감소나 실제 인구 비율을 뜻하지 않습니다.

| 조건을 일부 제거한 통제 | 형식화 기준과 판독이 다른 행 / 기준 판독 가능 행 |
|---|---:|
| 2022년 월세 예외 제거 | {controls['drop_exception']['disagreements']} / {len(known)} |
| 2023년에도 2022년 급여 제외 목록 사용 | {controls['keep_old_welfare']['disagreements']} / {len(known)} |

이 비교는 동일한 수동 규칙에서 조건을 제거하는 민감도 점검입니다. 신경망 비교나 독립 정답에 대한 정확도 측정이 아닙니다.
기준 자체가 보류한 행은 통제 불일치 분모에서 제외하고 전체 표에는 남겼습니다.

## 해석이 바뀐 지점

- 월세 예외를 생략하면 이전부터 조건을 충족하던 사례가 새로 포함된 것처럼 보일 수 있습니다.
- 주거 조건의 확대와 복지급여 제외 목록의 변화는 반대 방향으로 작동할 수 있습니다.
- 2022년 환산 문구를 1천원 단위 내림으로 읽은 후보와 공고에 실린 구간표가 다른 답을 주면 자동으로 한쪽을 선택하지 않습니다. 이는 공식 오류 판정이 아닙니다.
- 근거를 가리거나 입력을 모르면 그 조건은 미확인입니다. 별도의 명시적 제외 조건이 성립하면 AND 결론은 미충족일 수 있습니다.
- 최소 변경 원인은 두 조건 묶음의 부분집합 중 결론을 뒤집는 포함관계상 최소 집합입니다. 자연어 설명의 최단성이나 실제 인과효과가 아닙니다.

## 재실행

`python3 -m unittest -v test_engine.py`

`python3 run.py --output /tmp/policy-semantic-diff-replay`

`summary.json`, `scenarios.json`, `grid.csv`, `grid.jsonl`, `RESULTS.md`가 결정적으로 다시 생성됩니다.
원본 문서를 다시 받을 때는 sources.json의 내용 해시와 공고번호를 함께 확인해야 합니다.
"""
    (output / "RESULTS.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "counts": counts,
                "controls": {
                    k: v.get("disagreements", v.get("missed_direction_changes"))
                    for k, v in controls.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    run(parser.parse_args().output)
