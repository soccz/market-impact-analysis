"""
C-1 채널별 NLI sanity — 4정책 × 6채널 매트릭스
==============================================
질문: 6채널 원자가설이 실제 기사에서 서로 다른 정책 전달경로를 분리해내는가?

대표 4개 정책:
  pn=3  일본 수출규제 (E_foreign_shock, D=-) — 기대: C3, C4 높음
  pn=8  K칩스법 1차   (A_tax_support, D=+)   — 기대: C1, C5 높음 (+C3, C6도 가능)
  pn=15 공매도금지 3차 (B_market_stabilizer, D=+) — 기대: C5, C6 높음 (표면 감성 어려움)
  pn=28 8·2 부동산대책 (B_restriction_negative, D=-) — 기대: C2, C4 높음

각 정책 relevance ≥0.4 통과 기사 6건씩.
multi_label=True NLI로 6채널 가설 독립 추론.

평가 기준 (sanity, 성능평가 아님):
1. dominant channel이 정책 직관과 대체로 일치
2. 무관 채널이 전부 높게 뜨지 않음
3. 공매도 금지에서 C5/C6 분리 (표면감성 우회)
4. 일본 수출규제에서 C3/C4 분리
"""
import csv, re, json
import openpyxl
from collections import defaultdict
from transformers import pipeline

# ─────────────────────────────────────────
# 정책 카드 v0.2 로드
# ─────────────────────────────────────────
with open('policy_cards_v0.2.csv', encoding='utf-8-sig') as f:
    cards = {int(c['pn']): c for c in csv.DictReader(f)}

# top_stocks
wb = openpyxl.load_workbook('50_정책마스터.xlsx', read_only=True, data_only=True)
ws = wb['1_50정책_마스터']
TOP_STOCKS = {}
KW = {}
for r in list(ws.iter_rows(values_only=True))[3:]:
    if not r or not r[0]: continue
    pn = int(r[0])
    TOP_STOCKS[pn] = r[9] or ''
    KW[pn] = {'kw1': r[10] or '', 'kw2': r[11] or '', 'not_words': r[13] or ''}
wb.close()

# ─────────────────────────────────────────
# 채널 가설 템플릿 (D_i 방향 1개씩, 6채널)
# ─────────────────────────────────────────
CHANNEL_PHRASES = {
    'C1': {'+': '세부담 감소와 자금 여력 개선',     '-': '세부담 증가와 비용 부담 확대'},
    'C2': {'+': '수요 확대와 매출 증가',             '-': '매출 위축과 거래 감소'},
    'C3': {'+': '설비투자 확대와 공급망 강화',       '-': '공급망 충격과 생산 차질'},
    'C4': {'+': '규제 완화와 정책 불확실성 해소',     '-': '규제 강화와 정책 불확실성 확대'},
    'C5': {'+': '유동성 공급과 자금 조달 개선',       '-': '유동성 위축과 자금 경색'},
    'C6': {'+': '시장 신뢰 회복과 투자심리 개선',     '-': '시장 신뢰 훼손과 투자심리 위축'},
}

INDUSTRY_MAP = {
    '반도체': '반도체 업종', '금융/증권': '금융 업종', '부동산/건설': '건설 업종',
    '바이오/제약': '제약·바이오 업종', '2차전지/에너지': '2차전지 업종',
}

def build_channel_hypotheses(card, top_stocks):
    D_i = card['D_i_initial']
    industry = INDUSTRY_MAP.get(card['industry'], card['industry'])
    stocks = ', '.join([s.strip() for s in top_stocks.split(',')[:2]
                        if s.strip() and s.strip() != '전 종목']) or industry
    direction = '상승' if D_i == '+' else '하락'
    hyps = {}
    for c in ['C1','C2','C3','C4','C5','C6']:
        phrase = CHANNEL_PHRASES[c][D_i]
        hyps[c] = f"이 기사는 '{card['alias']}' 정책이 {phrase}을 통해 {stocks} 등 {industry}의 주가에 {direction} 압력을 만든다고 설명한다"
    return hyps

# ─────────────────────────────────────────
# relevance v3.1 (sanity_check_v3_1.py 동일)
# ─────────────────────────────────────────
def parse_or(s):
    if not s or str(s).lower() == 'none': return []
    return [t.strip() for t in re.split(r'\s+OR\s+', str(s).strip())]
def norm(s):
    return re.sub(r'\s+', '', (s or '').lower())
def is_valid_token(t):
    return len(t) >= 2 and not t.isdigit()

def relevance_v3_1(title, content, kw):
    raw = (title or '') + ' ' + (content or '')
    t = norm(raw)
    for nw in parse_or(kw['not_words']):
        if nw and norm(nw) in t: return 0.0
    pri = parse_or(kw['kw1'])
    sec = parse_or(kw['kw2'])
    for k in pri:
        if k and norm(k) in t: return 1.0
    for k in sec:
        if k and norm(k) in t: return 0.7
    for k in pri:
        toks = [x for x in re.split(r'[\s·\-]+', k) if x and is_valid_token(x)]
        if len(toks) >= 2 and all(norm(x) in t for x in toks): return 0.7
    for k in sec:
        toks = [x for x in re.split(r'[\s·\-]+', k) if x and is_valid_token(x)]
        if len(toks) >= 2 and all(norm(x) in t for x in toks): return 0.5
    return 0.0

# ─────────────────────────────────────────
# 기사 샘플링: 각 정책에서 relevance≥0.4 통과 기사 6건
# ─────────────────────────────────────────
TARGET_PNS = [3, 8, 15, 28]
EXPECTED = {
    3:  {'high': ['C3', 'C4'], 'desc': '일본 수출규제 — 공급망·불확실성'},
    8:  {'high': ['C1', 'C5'], 'desc': 'K칩스법 1차 — 비용·유동성'},
    15: {'high': ['C5', 'C6'], 'desc': '공매도금지 3차 — 유동성·심리 (표면감성 우회)'},
    28: {'high': ['C2', 'C4'], 'desc': '8·2 대책 — 수요·규제'},
}

print(f"[1/4] 기사 샘플링 ({len(TARGET_PNS)}정책 × 6건)")
samples = defaultdict(list)
with open('50_전체기사_99539건.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        try: pn = int(row['policy_n'])
        except: continue
        if pn not in TARGET_PNS: continue
        if len(samples[pn]) >= 30: continue  # 후보 풀
        try: rd = int(row['rel_day'])
        except: continue
        if not (-7 <= rd <= 5): continue  # 메인창 안
        samples[pn].append(row)

# 정책별 relevance 통과 기사 6건 (rel_day 0 근처 우선)
selected = {}
for pn in TARGET_PNS:
    kw = KW[pn]
    rated = []
    for r in samples[pn]:
        rel = relevance_v3_1(r['title'], r['content'], kw)
        if rel >= 0.4:
            rated.append((rel, abs(int(r['rel_day'])), r))
    rated.sort(key=lambda x: (-x[0], x[1]))  # rel 높고 정책일 가까운 순
    selected[pn] = [t[2] for t in rated[:6]]
    print(f"  pn={pn}: 후보 {len(samples[pn])}건 중 relevance≥0.4 {len(rated)}건, 상위 {len(selected[pn])}건 선별")

print(f"\n[2/4] NLI 모델 로드")
zsc = pipeline("zero-shot-classification",
               model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=-1)
print("  ✓")

print(f"\n[3/4] 6채널 multi_label NLI 추론")

results = []
for pn in TARGET_PNS:
    card = cards[pn]
    top = TOP_STOCKS[pn]
    hyps = build_channel_hypotheses(card, top)
    active = [c for c in ['C1','C2','C3','C4','C5','C6'] if card.get(c)=='ON']
    exp = EXPECTED[pn]['high']
    print(f"\n--- pn={pn} {card['alias']} ({card['type_subtype']}, D={card['D_i_initial']})")
    print(f"    active channels: {active}  |  expected high: {exp}")

    for art in selected[pn]:
        text = f"{art['title']}\n{art['content'] or ''}"
        cand = [hyps[c] for c in ['C1','C2','C3','C4','C5','C6']]
        out = zsc(text, cand, multi_label=True)
        scores = dict(zip(out['labels'], out['scores']))
        ch_scores = {c: round(scores[hyps[c]], 3) for c in ['C1','C2','C3','C4','C5','C6']}
        dominant = max(ch_scores, key=ch_scores.get)
        results.append({
            'pn': pn, 'title': art['title'][:40], 'rel_day': art['rel_day'],
            **ch_scores, 'dominant': dominant,
        })
        # 컬럼별 출력 (active는 *표시)
        line = f"    [D{int(art['rel_day']):+d}] "
        for c in ['C1','C2','C3','C4','C5','C6']:
            mark = '*' if c in active else ' '
            v = ch_scores[c]
            line += f"{c}{mark}={v:.2f} "
        line += f"→ dom={dominant}"
        print(line)
        print(f"            \"{art['title'][:55]}\"")

print(f"\n[4/4] 분석 — 정책별 평균 채널 점수")
print("-" * 95)
print(f"{'pn':>3s} {'정책':<18s} {'expected':<12s} | C1   C2   C3   C4   C5   C6   | dominant 빈도")
print("-" * 95)
for pn in TARGET_PNS:
    card = cards[pn]
    pn_rows = [r for r in results if r['pn'] == pn]
    avg = {c: sum(r[c] for r in pn_rows) / len(pn_rows) for c in ['C1','C2','C3','C4','C5','C6']}
    dom_counter = defaultdict(int)
    for r in pn_rows: dom_counter[r['dominant']] += 1
    dom_str = ', '.join(f"{k}:{v}" for k, v in sorted(dom_counter.items(), key=lambda x: -x[1]))
    exp = '/'.join(EXPECTED[pn]['high'])
    avg_str = ' '.join(f"{avg[c]:.2f}" for c in ['C1','C2','C3','C4','C5','C6'])
    print(f"{pn:>3d} {card['alias']:<18s} {exp:<12s} | {avg_str} | {dom_str}")
print("-" * 95)

# 합격 판정 (sanity)
print(f"\n=== Sanity 판정 ===")
for pn in TARGET_PNS:
    pn_rows = [r for r in results if r['pn'] == pn]
    avg = {c: sum(r[c] for r in pn_rows) / len(pn_rows) for c in ['C1','C2','C3','C4','C5','C6']}
    exp_high = EXPECTED[pn]['high']
    # Top 2 채널이 expected와 얼마나 겹치는가
    top2 = sorted(avg.items(), key=lambda x: -x[1])[:2]
    top2_set = {c for c, _ in top2}
    overlap = top2_set & set(exp_high)
    if len(overlap) >= 2:
        verdict = '✓ Top2 일치'
    elif len(overlap) == 1:
        verdict = '~ Top2 1개 일치'
    else:
        verdict = '✗ Top2 불일치'
    top2_str = ','.join(f"{c}({v:.2f})" for c, v in top2)
    print(f"  pn={pn} {EXPECTED[pn]['desc']}")
    print(f"    expected: {exp_high}  |  observed Top2: {top2_str}  →  {verdict}")

with open('sanity_C_channels_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n저장: sanity_C_channels_results.json")
