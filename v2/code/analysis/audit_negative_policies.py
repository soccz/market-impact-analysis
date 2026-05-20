#!/usr/bin/env python3
"""
악재 정책(direction=-) 17건 정성 진단.

목적:
- 24% 일치율이 모델 한계인지, 데이터 한계인지, 자연적 약함인지 판별

산출:
- reports/negative_audit.md (정책별 카드 + 의심 케이스)
- reports/negative_audit.html (한 화면 시각화)
- data/processed/negative_policy_audit.csv (정량 지표)
"""
import csv
import html
from pathlib import Path

import numpy as np
import pandas as pd


V1_PATH = "data/processed/article_scores_hybrid.csv"
POLICY_CARDS = "policy_cards_v1.1.csv"
REPORT_MD = "reports/negative_audit.md"
REPORT_HTML = "reports/negative_audit.html"
AUDIT_CSV = "data/processed/negative_policy_audit.csv"


def load_policy_cards():
    cards = {}
    with open(POLICY_CARDS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cards[int(r["pn"])] = r
    return cards


def main():
    cards = load_policy_cards()
    neg_pns = sorted([pn for pn, c in cards.items() if c["D_i_initial"] == "-"])
    print(f"악재 정책(direction=-): {len(neg_pns)}건 — {neg_pns}")

    v1 = pd.read_csv(V1_PATH, encoding="utf-8-sig", low_memory=False)
    v1["pn"] = pd.to_numeric(v1["pn"], errors="coerce")
    v1 = v1.dropna(subset=["pn"]).copy()
    v1["pn"] = v1["pn"].astype(int)
    v1["rel_day"] = pd.to_numeric(v1["rel_day"], errors="coerce")
    v1["PTEI_total"] = pd.to_numeric(v1["PTEI_total"], errors="coerce").fillna(0)
    v1["Contradiction_total"] = pd.to_numeric(v1["Contradiction_total"], errors="coerce").fillna(0)
    v1["p_support"] = pd.to_numeric(v1["p_support"], errors="coerce").fillna(0)
    v1["p_contradict"] = pd.to_numeric(v1["p_contradict"], errors="coerce").fillna(0)

    WIN_LO, WIN_HI = -7, 5
    win = v1[(v1["pn"].isin(neg_pns)) & (v1["rel_day"] >= WIN_LO) & (v1["rel_day"] <= WIN_HI)].copy()
    print(f"악재 정책 윈도우 기사: {len(win):,}건")

    # 정책별 진단
    audit_rows = []
    for pn in neg_pns:
        card = cards[pn]
        sub = win[win["pn"] == pn].copy()
        if len(sub) == 0:
            continue
        ptei_sum = sub["PTEI_total"].sum()
        contra_sum = sub["Contradiction_total"].sum()
        ptei_mean = sub["PTEI_total"].mean()
        contra_mean = sub["Contradiction_total"].mean()
        # 정책이 악재이므로 contradict > PTEI 가 일치 신호
        signal = contra_mean - ptei_mean
        aligned = signal > 0
        # 의심 — PTEI가 contra의 2배 이상이면 D_i 의심
        suspicious = ptei_sum > 2 * contra_sum and contra_sum > 0

        # Top 5 PTEI 기사 (호재 톤으로 잡힌)
        top_ptei = sub.nlargest(5, "PTEI_total")[["title", "pub_date", "rel_day", "PTEI_total", "Contradiction_total", "p_support", "p_contradict"]]
        # Top 5 Contradict 기사 (악재 톤으로 잡힌)
        top_contra = sub.nlargest(5, "Contradiction_total")[["title", "pub_date", "rel_day", "PTEI_total", "Contradiction_total", "p_support", "p_contradict"]]

        audit_rows.append({
            "pn": pn,
            "alias": card["alias"],
            "industry": card["industry"],
            "type_subtype": card["type_subtype"],
            "n_articles": len(sub),
            "ptei_mean": round(ptei_mean, 3),
            "contra_mean": round(contra_mean, 3),
            "signal": round(signal, 3),
            "aligned": aligned,
            "suspicious_d_i": suspicious,
            "top_ptei": top_ptei,
            "top_contra": top_contra,
        })

    # CSV 저장 (top_* 제외)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("top_ptei", "top_contra")} for r in audit_rows])
    df.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved: {AUDIT_CSV}")

    aligned_count = int(df["aligned"].sum())
    susp_count = int(df["suspicious_d_i"].sum())
    print(f"\n악재 정책 일치: {aligned_count}/{len(df)} ({aligned_count/len(df):.1%})")
    print(f"D_i 의심 (PTEI >> Contra): {susp_count}건")

    # ===== HTML 보고서 =====
    html_parts = ['<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>악재 정책 진단</title>',
                  '<style>',
                  'body { font-family: -apple-system, BlinkMacSystemFont, AppleGothic, sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 20px; color: #222; line-height: 1.55; }',
                  'h1 { font-size: 22px; }',
                  'h2 { font-size: 18px; margin-top: 28px; }',
                  '.summary { background: #f0f7ff; border: 1px solid #b3d4fc; padding: 14px 18px; border-radius: 6px; margin: 16px 0; }',
                  '.card { border: 1px solid #ddd; border-radius: 8px; padding: 14px 18px; margin: 16px 0; background: #fafafa; }',
                  '.card-head { display: flex; justify-content: space-between; align-items: baseline; font-size: 14px; color: #555; }',
                  '.policy { font-weight: 600; color: #1a73e8; font-size: 16px; }',
                  '.aligned { color: #137333; font-weight: 600; }',
                  '.misaligned { color: #c5221f; font-weight: 600; }',
                  '.suspicious { background: #fff3cd; padding: 2px 6px; border-radius: 3px; color: #856404; font-weight: 600; }',
                  'table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }',
                  'th, td { border: 1px solid #e0e0e0; padding: 4px 8px; text-align: left; vertical-align: top; }',
                  'th { background: #f0f0f0; font-weight: 600; }',
                  '.title-cell { max-width: 380px; }',
                  '.bar { display: inline-block; height: 12px; background: #1a73e8; vertical-align: middle; }',
                  '.bar-red { background: #d93025; }',
                  '</style></head><body>',
                  f'<h1>악재 정책 17건 정성 진단</h1>',
                  f'<div class="summary">',
                  f'<b>전체 일치율</b>: {aligned_count}/{len(df)} ({aligned_count/len(df):.1%})<br>',
                  f'<b>D_i 의심 케이스</b> (PTEI >> Contradict): {susp_count}건<br>',
                  f'<b>판독 가이드</b>: PTEI 막대(파랑)가 Contradict(빨강)보다 길면 → 뉴스가 정책을 호재로 보도 → 정책 방향(-)과 모순<br>',
                  f'PTEI ≈ Contradict면 → 보도가 양면 또는 사실 보도 중심<br>',
                  f'Contradict가 길면 → 뉴스가 정책에 비판적 (D_i=- 일치)',
                  f'</div>']

    # 정책 카드들
    audit_rows_sorted = sorted(audit_rows, key=lambda x: -x["signal"])  # 일치 강한 것 위로
    for r in audit_rows_sorted:
        cls = "aligned" if r["aligned"] else "misaligned"
        align_text = "✅ 일치 (D_i=- 따라감)" if r["aligned"] else "❌ 불일치 (호재처럼 보도됨)"
        susp_tag = '<span class="suspicious">⚠️ D_i 의심 — PTEI 신호가 너무 강함</span>' if r["suspicious_d_i"] else ""

        # 막대 그래프 (PTEI vs Contra)
        max_v = max(r["ptei_mean"], r["contra_mean"], 0.001)
        ptei_w = int(r["ptei_mean"] / max_v * 200)
        contra_w = int(r["contra_mean"] / max_v * 200)

        html_parts.append(f'<div class="card">')
        html_parts.append(f'<div class="card-head"><span><span class="policy">pn={r["pn"]} · {html.escape(r["alias"])}</span> · {html.escape(r["industry"])} · {html.escape(r["type_subtype"])}</span><span class="{cls}">{align_text}</span></div>')
        html_parts.append(f'<div style="margin-top:8px; font-size:13px;">기사 {r["n_articles"]}건 · PTEI 평균 <b>{r["ptei_mean"]:.3f}</b> · Contradict 평균 <b>{r["contra_mean"]:.3f}</b> · 신호={r["signal"]:+.3f} {susp_tag}</div>')
        html_parts.append(f'<div style="margin-top:6px; font-size:11px;">PTEI: <span class="bar" style="width:{ptei_w}px"></span> {r["ptei_mean"]:.3f}<br>Contra: <span class="bar bar-red" style="width:{contra_w}px"></span> {r["contra_mean"]:.3f}</div>')

        # Top PTEI 기사 (호재 톤)
        html_parts.append(f'<details><summary style="cursor:pointer; margin-top:8px; font-size:13px;">▶ Top 5 PTEI 기사 (호재 톤으로 잡힌 — 모순일 가능성)</summary>')
        html_parts.append('<table><thead><tr><th>날짜</th><th>D±</th><th>제목</th><th>PTEI</th><th>Contra</th><th>p_sup</th><th>p_con</th></tr></thead><tbody>')
        for _, row in r["top_ptei"].iterrows():
            html_parts.append(f'<tr><td>{row["pub_date"]}</td><td>{int(row["rel_day"]) if pd.notna(row["rel_day"]) else ""}</td><td class="title-cell">{html.escape(str(row["title"])[:100])}</td><td>{row["PTEI_total"]:.3f}</td><td>{row["Contradiction_total"]:.3f}</td><td>{row["p_support"]:.2f}</td><td>{row["p_contradict"]:.2f}</td></tr>')
        html_parts.append('</tbody></table></details>')

        # Top Contradict
        html_parts.append(f'<details><summary style="cursor:pointer; margin-top:8px; font-size:13px;">▶ Top 5 Contradict 기사 (악재 톤으로 잡힌 — D_i=- 일치)</summary>')
        html_parts.append('<table><thead><tr><th>날짜</th><th>D±</th><th>제목</th><th>PTEI</th><th>Contra</th><th>p_sup</th><th>p_con</th></tr></thead><tbody>')
        for _, row in r["top_contra"].iterrows():
            html_parts.append(f'<tr><td>{row["pub_date"]}</td><td>{int(row["rel_day"]) if pd.notna(row["rel_day"]) else ""}</td><td class="title-cell">{html.escape(str(row["title"])[:100])}</td><td>{row["PTEI_total"]:.3f}</td><td>{row["Contradiction_total"]:.3f}</td><td>{row["p_support"]:.2f}</td><td>{row["p_contradict"]:.2f}</td></tr>')
        html_parts.append('</tbody></table></details>')

        html_parts.append('</div>')

    html_parts.append('</body></html>')
    Path(REPORT_HTML).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"saved: {REPORT_HTML}")

    # ===== Markdown 요약 보고서 =====
    md = [f"# 악재 정책 17건 정성 진단\n",
          f"## 요약\n",
          f"- 일치 정책: **{aligned_count}/{len(df)} ({aligned_count/len(df):.1%})**",
          f"- D_i 의심 케이스: {susp_count}건",
          f"- 윈도우: D{WIN_LO}~D+{WIN_HI}\n",
          f"## 정책별 진단 (일치 신호 순)\n",
          "| pn | 정책 | 산업 | type | n_articles | PTEI_mean | Contra_mean | signal | aligned | 의심 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in audit_rows_sorted:
        align_emoji = "✅" if r["aligned"] else "❌"
        susp = "⚠️" if r["suspicious_d_i"] else ""
        md.append(f"| {r['pn']} | {r['alias']} | {r['industry']} | {r['type_subtype']} | {r['n_articles']} | {r['ptei_mean']:.3f} | {r['contra_mean']:.3f} | {r['signal']:+.3f} | {align_emoji} | {susp} |")

    md.append("\n## 진단 카테고리\n")
    md.append("### A. 모델이 정확히 잡은 케이스 (signal > 0)")
    for r in audit_rows_sorted:
        if r["aligned"]:
            md.append(f"- **pn={r['pn']} {r['alias']}**: Contra가 PTEI 보다 큼 → 정상")
    md.append("\n### B. 모델이 못 잡았지만 자연스러운 케이스 (signal ≈ 0)")
    for r in audit_rows_sorted:
        if not r["aligned"] and not r["suspicious_d_i"]:
            md.append(f"- **pn={r['pn']} {r['alias']}**: PTEI {r['ptei_mean']:.2f} ≈ Contra {r['contra_mean']:.2f} → 양면 보도")
    md.append("\n### C. D_i 분류 의심 케이스 (PTEI >> Contra, signal << 0)")
    for r in audit_rows_sorted:
        if r["suspicious_d_i"]:
            md.append(f"- **pn={r['pn']} {r['alias']}**: PTEI {r['ptei_mean']:.2f} vs Contra {r['contra_mean']:.2f} → 호재처럼 보도됨. D_i=-가 정말 맞나?")

    md.append(f"\n## 자세히 보기\n- HTML: [reports/negative_audit.html](negative_audit.html)\n- 정량 CSV: [data/processed/negative_policy_audit.csv](../data/processed/negative_policy_audit.csv)\n")

    Path(REPORT_MD).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"saved: {REPORT_MD}")

    print(f"\n=== 결론 미리보기 ===")
    cat_a = sum(1 for r in audit_rows_sorted if r["aligned"])
    cat_c = sum(1 for r in audit_rows_sorted if r["suspicious_d_i"])
    cat_b = len(audit_rows_sorted) - cat_a - cat_c
    print(f"A 모델이 정확히 잡음 (aligned): {cat_a}건")
    print(f"B PTEI≈Contra 자연스러운 양면 보도: {cat_b}건")
    print(f"C D_i 분류 의심 (PTEI>>Contra): {cat_c}건")


if __name__ == "__main__":
    main()
