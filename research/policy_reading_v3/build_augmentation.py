"""Post-hoc factorial audit of amount-format and irrelevant-event shortcuts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from build_data import amounts

ROOT = Path(__file__).resolve().parent


def rewrite_amounts(row):
    row = deepcopy(row)
    # Each role appears in each format. Templates and relation/state labels unchanged.
    base = (row["frame"] + row["family"]) % 3
    for span in sorted(row["spans"], key=lambda s: s["start"], reverse=True):
        offset = {"before": 0, "after": 1, "other": 2}[span["role"]]
        style = (base + offset) % 3
        won = int(span["won"])
        value = [f"{won//10000}만 원", f"{won:,}원", f"{won//1000:,}천 원"][style]
        end = span["end"]
        delta = len(value) - (end - span["start"])
        row["text"] = row["text"][: span["start"]] + value + row["text"][end:]
        for other in row["spans"] + row["evidence"]:
            if other is not span and other["start"] >= end:
                other["start"] += delta
                other["end"] += delta
        span["end"] = span["start"] + len(value)
        span["text"] = value
    return row


def insert_distractor(row):
    row = deepcopy(row)
    # Deterministic counterbalancing independent of target relation and state.
    ri = (row["frame"] + row["family"]) % 4
    si = (row["frame"] * 3 + row["family"]) % 6
    rel = ["금액 정정", "지원 규모 변경", "동일 액수 재표기", "사유 없는 금액 병기"][ri]
    state = [
        "확정되었다",
        "진행될 예정이다",
        "검토 중이다",
        "있었다는 주장은 사실이 아니다",
        "있었다고 가정한다",
        "상태가 밝혀지지 않았다",
    ][si]
    first = 210 + row["frame"]
    second = first if ri == 2 else first + 35
    a = f"{first}만원"
    b = f"{second*10000:,}원" if ri == 2 else f"{second}만원"
    clause = f"별빛사업의 {rel} 사안은 {state}. 참고 금액: {a}, {b}. "
    extras = [dict(x, role="other") for x in amounts(clause)]
    if (row["family"] + row["frame"]) % 2:
        for x in row["spans"] + row["evidence"]:
            x["start"] += len(clause)
            x["end"] += len(clause)
        row["text"] = clause + row["text"]
        row["spans"] = extras + row["spans"]
    else:
        prefix = len(row["text"]) + 1
        for x in extras:
            x["start"] += prefix
            x["end"] += prefix
        row["text"] += " " + clause
        row["spans"] += extras
    return row


def main():
    assert not (ROOT / "augmentation_freeze.json").exists()
    original = [
        json.loads(s) for s in (ROOT / "data/train.jsonl").read_text().splitlines()
    ]
    out = ROOT / "augmentation_data"
    out.mkdir(exist_ok=True)
    for condition in ["units", "scope", "both"]:
        rows = []
        for row in original:
            x = (
                rewrite_amounts(row)
                if condition in ["units", "both"]
                else deepcopy(row)
            )
            if condition in ["scope", "both"]:
                x = insert_distractor(x)
            x.update(
                id=hashlib.sha256((row["id"] + condition).encode()).hexdigest()[:16],
                condition="augmentation_" + condition,
            )
            for sp in x["spans"] + x["evidence"]:
                assert x["text"][sp["start"] : sp["end"]] == sp["text"]
            assert [
                {k: v for k, v in sp.items() if k != "role"} for sp in x["spans"]
            ] == amounts(x["text"])
            assert [
                (sp["role"], sp["won"]) for sp in x["spans"] if sp["role"] != "other"
            ] == [
                (sp["role"], sp["won"]) for sp in row["spans"] if sp["role"] != "other"
            ]
            rows.append(x)
        (out / f"{condition}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        )
    print(
        "Three training-only variants, 960 rows each; target labels and 4 excluded compositions preserved"
    )


if __name__ == "__main__":
    main()
