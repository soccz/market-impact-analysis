#!/usr/bin/env python3
"""
B 케이스 39건 (low confidence 2명 이상) 검토용 페이지 생성.

산출:
- splits/review_B_39.html        : 한 화면에 카드 39개 (시각화)
- splits/review_B_39_input.csv   : 본인이 결정 채워넣을 입력 파일
                                   (your_relevance, your_stance, your_note 컬럼)
- 입력 채운 후: apply_review_B_39.py 실행하면 train_500_final.csv 자동 업데이트
"""
import csv
import html


INPUT = "splits/train_500_consensus.csv"
OUT_HTML = "review_B_39.html"
OUT_CSV = "review_B_39_input.csv"


def main():
    with open(INPUT, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # B 케이스: 불일치(needs_review_low) + low confidence ≥ 2
    B_rows = []
    for r in rows:
        if r.get("consensus_status") != "needs_review_low":
            continue
        confs = [r["conf_1"], r["conf_2"], r["conf_3"]]
        if confs.count("low") >= 2:
            B_rows.append(r)

    print(f"B 케이스 추출: {len(B_rows)}건 (불일치 + low 2명 이상)")

    # 입력용 CSV (본인이 채울 곳)
    csv_cols = ["news_id", "pn", "alias", "direction", "title_short",
                "auto_relevance", "auto_stance",
                "your_relevance", "your_stance", "your_note"]
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in B_rows:
            from collections import Counter
            rels = [r["rel_1"], r["rel_2"], r["rel_3"]]
            stances = [r["stance_1"], r["stance_2"], r["stance_3"]]
            auto_rel = Counter(rels).most_common(1)[0][0]
            auto_stance = Counter(stances).most_common(1)[0][0]
            w.writerow({
                "news_id": r["news_id"],
                "pn": r["pn"],
                "alias": r["alias"],
                "direction": r["direction"],
                "title_short": r["title"][:60],
                "auto_relevance": auto_rel,
                "auto_stance": auto_stance,
                "your_relevance": "",
                "your_stance": "",
                "your_note": "",
            })

    # HTML 시각화
    html_parts = ["""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>B 케이스 39건 검토</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 20px; color: #222; line-height: 1.5; }
h1 { font-size: 22px; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; background: #fafafa; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; font-size: 14px; color: #555; margin-bottom: 8px; }
.policy { font-weight: 600; color: #1a73e8; }
.dir { display: inline-block; padding: 1px 6px; border-radius: 3px; font-weight: 700; }
.dir-plus { background: #e0f0ff; color: #0050b3; }
.dir-minus { background: #ffeaea; color: #b30000; }
.title { font-size: 16px; font-weight: 600; margin: 6px 0; }
.meta { font-size: 12px; color: #888; }
.content { font-size: 13px; color: #333; background: white; border-left: 3px solid #ddd; padding: 8px 12px; margin: 10px 0; max-height: 200px; overflow-y: auto; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
th, td { border: 1px solid #e0e0e0; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; font-weight: 600; }
.auto { background: #fffae6; padding: 8px 12px; border-radius: 4px; margin-top: 10px; font-size: 13px; }
.low { background: #ffe9e9; }
</style></head><body>
<h1>B 케이스 39건 — 본인 검토 화면</h1>
<p><b>이용법</b>: 각 카드 보고 자동 다수결과 다른 결정이면 <code>splits/review_B_39_input.csv</code>의 <code>your_relevance</code>, <code>your_stance</code> 컬럼에 입력. 그대로 두면 자동 다수결 유지.</p>
"""]

    for idx, r in enumerate(B_rows, start=1):
        from collections import Counter
        rels = [r["rel_1"], r["rel_2"], r["rel_3"]]
        stances = [r["stance_1"], r["stance_2"], r["stance_3"]]
        confs = [r["conf_1"], r["conf_2"], r["conf_3"]]
        reasons = [r.get("reason_1", ""), r.get("reason_2", ""), r.get("reason_3", "")]
        auto_rel = Counter(rels).most_common(1)[0][0]
        auto_stance = Counter(stances).most_common(1)[0][0]

        dir_class = "dir-plus" if r["direction"] == "+" else "dir-minus"
        dir_label = "+ 호재/지원" if r["direction"] == "+" else "- 규제/충격"

        rows_html = ""
        for i in range(3):
            low_class = ' class="low"' if confs[i] == "low" else ""
            rows_html += (
                f'<tr{low_class}><td>{i+1}</td>'
                f'<td>{html.escape(str(rels[i]))}</td>'
                f'<td>{html.escape(stances[i])}</td>'
                f'<td>{html.escape(confs[i])}</td>'
                f'<td>{html.escape(reasons[i])}</td></tr>'
            )

        card = f"""
<div class="card">
<div class="card-head">
  <span>[{idx}/{len(B_rows)}] news_id={html.escape(r['news_id'])}</span>
  <span><span class="policy">pn={r['pn']} · {html.escape(r['alias'])}</span> · <span class="dir {dir_class}">{dir_label}</span></span>
</div>
<div class="title">{html.escape(r['title'])}</div>
<div class="meta">{r['pub_date']} | {html.escape(r['provider'])} | {html.escape(r['industry'])} | rel_day={r['rel_day']}</div>
<div class="content">{html.escape(r['content'][:1000])}{'…' if len(r['content']) > 1000 else ''}</div>
<table>
  <thead><tr><th>라벨러</th><th>relevance</th><th>stance</th><th>confidence</th><th>reason</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<div class="auto">⚙️ 자동 다수결: <b>relevance={auto_rel}, stance={auto_stance}</b> &nbsp;|&nbsp; CSV에 다른 값 적으면 그걸로 덮어씀</div>
</div>
"""
        html_parts.append(card)

    html_parts.append("</body></html>")

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"saved: {OUT_HTML}")
    print(f"saved: {OUT_CSV}")
    print(f"\n사용법:")
    print(f"  1) open {OUT_HTML}  ← 브라우저로 열어 한 화면에 39건 보기")
    print(f"  2) {OUT_CSV}의 your_* 컬럼에 결정 입력 (빈칸 = 자동 다수결 유지)")
    print(f"  3) python apply_review_B_39.py  ← 적용 (다음 단계에서 만들 예정)")


if __name__ == "__main__":
    main()
