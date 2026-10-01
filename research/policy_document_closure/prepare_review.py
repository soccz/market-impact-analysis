"""Make local blinded review packets with no model outputs or reference labels."""

import argparse
import csv
import html
import json
from pathlib import Path
import random

from rehydrate import reconstruct


def packet(data, seed):
    docs = {d["id"]: d for d in data["documents"]}
    rows = list(data["cases"])
    random.Random(seed).shuffle(rows)
    return [
        dict(
            review_id=f"item-{i+1:03d}",
            case_id=c["id"],
            document=c["document"],
            scope=docs[c["document"]]["scope"],
            fields=docs[c["document"]]["fields"],
            facts=c["facts"],
            evidence=docs[c["document"]]["evidence"],
        )
        for i, c in enumerate(rows)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = reconstruct(args.sources)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for reviewer, seed in [("A", 20261002), ("B", 20261003)]:
        path = out / reviewer
        path.mkdir(exist_ok=True)
        records = packet(data, seed)
        (path / "packet.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n"
        )
        with (path / "answers.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "review_id",
                    "case_id",
                    "reviewer",
                    "independent_label",
                    "evidence",
                    "uncertainty",
                    "seconds",
                ]
            )
            for r in records:
                writer.writerow([r["review_id"], r["case_id"], "", "", "", "", ""])
        body = [
            '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>독립 판독 자료</title><style>body{max-width:850px;margin:30px auto;padding:16px;font:16px/1.8 sans-serif;color:#253e48}section{border-top:2px solid #9ab5c0;margin:38px 0;padding-top:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f6f7;padding:15px}summary{cursor:pointer}</style><h1>독립 판독 자료</h1><p>판정 범위와 사실, 원문을 읽고 answers.csv에 판정과 근거를 기록하세요. 정답·다른 사람의 판독·모델 결과는 포함하지 않았습니다. 원문은 검토용 로컬 자료이며 재배포하지 않습니다.</p>'
        ]
        for r in records:
            esc = html.escape
            body.append(
                "<section><h2>"
                + esc(r["review_id"])
                + "</h2><p>"
                + esc(r["scope"])
                + "</p><pre>"
                + esc(
                    json.dumps(
                        dict(fields=r["fields"], facts=r["facts"]),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre><details><summary>문서 본문 전체</summary><pre>"
                + esc("\n\n".join(k + "\n" + v for k, v in r["evidence"].items()))
                + "</pre></details></section>"
            )
        (path / "index.html").write_text("".join(body) + "</html>\n")
    print(
        "Prepared two different orders of the same 48 cases; all answer cells remain blank."
    )


if __name__ == "__main__":
    main()
