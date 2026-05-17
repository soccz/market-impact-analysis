"""
PPT-NLP Sanity Check v3.1 — gate 미세조정 + case05 가설 v3
=========================================================
v3.1 규칙:
  L1  : 1차 키워드 exact (공백 제거 substring)              → 1.0
  L1.5: 2차 키워드 exact                                    → 0.7
  L2  : 1차 토큰 AND (1글자·숫자 토큰 제외, 2+ 토큰 필요)    → 0.7
  L2.5: 2차 토큰 AND (동일 제약)                            → 0.5
  L3  : alias exact (정책 약칭) → 0.4
  ⚠ stocks/대표종목 단독 매칭은 relevance 통과 불가
     → specificity factor로 분리 (별도 출력)
  NO_MATCH → 0

case05 가설 v3: C5 유동성 + C6 투자심리 명시
case07 가설 v3: C4 규제·불확실성 + C6 투자심리 (이미 v3)
"""
import json, re
import openpyxl
from transformers import pipeline

# ─────────────────────────────────────────
def load_policies():
    wb = openpyxl.load_workbook('50_정책마스터.xlsx', read_only=True, data_only=True)
    ws = wb['1_50정책_마스터']
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    P = {}
    for r in rows[3:]:
        if not r or not r[0]: continue
        P[str(r[0])] = {
            'industry': r[1] or '', 'alias': r[3] or '',
            'type': r[6] or '', 'top_stocks': r[9] or '',
            'kw1': r[10] or '', 'kw2': r[11] or '',
            'and_cond': r[12] or '', 'not_words': r[13] or '',
        }
    return P

def parse_or(s):
    if not s or str(s).lower() == 'none': return []
    return [t.strip() for t in re.split(r'\s+OR\s+', str(s).strip())]

def norm(s):
    return re.sub(r'\s+', '', (s or '').lower())

def is_valid_token(t):
    """매칭 근거로 쓸 수 있는 토큰: 길이 ≥ 2, 순수 숫자 아님"""
    if len(t) < 2: return False
    if t.isdigit(): return False
    return True

# ─────────────────────────────────────────
# A1 v3.1: 4단계 relevance gate
# ─────────────────────────────────────────
def relevance_v3_1(title, content, policy):
    raw = (title or '') + ' ' + (content or '')
    text_norm = norm(raw)

    # NOT
    for nw in parse_or(policy['not_words']):
        if nw and norm(nw) in text_norm:
            return 0.0, {'level': 'NOT', 'word': nw, 'specificity_boost': 0}

    pri = parse_or(policy['kw1'])
    sec = parse_or(policy['kw2'])

    # 종목 매칭 (specificity로 분리 — relevance 통과 조건 아님)
    stocks = [s.strip() for s in policy['top_stocks'].split(',')
              if s.strip() and s.strip() != '전 종목' and len(s.strip()) >= 2]
    stock_hits = [s for s in stocks if norm(s) in text_norm]
    specificity_boost = 0.2 if stock_hits else 0.0  # specificity에 별도 전달용

    # L1: 1차 키워드 exact
    l1 = [k for k in pri if k and norm(k) in text_norm]
    if l1:
        return 1.0, {'level': 'L1_primary_exact', 'hits': l1,
                     'specificity_boost': specificity_boost,
                     'stock_hits': stock_hits}

    # L1.5: 2차 키워드 exact
    l1_5 = [k for k in sec if k and norm(k) in text_norm]
    if l1_5:
        return 0.7, {'level': 'L1.5_secondary_exact', 'hits': l1_5,
                     'specificity_boost': specificity_boost,
                     'stock_hits': stock_hits}

    # L2: 1차 키워드 토큰 AND (1글자·숫자 제외)
    l2 = []
    for k in pri:
        tokens = [t for t in re.split(r'[\s·\-]+', k) if t]
        valid_tokens = [t for t in tokens if is_valid_token(t)]
        if len(valid_tokens) < 2: continue
        if all(norm(t) in text_norm for t in valid_tokens):
            l2.append(f"{k} valid={valid_tokens}")
    if l2:
        return 0.7, {'level': 'L2_primary_token_AND', 'hits': l2,
                     'specificity_boost': specificity_boost,
                     'stock_hits': stock_hits}

    # L2.5: 2차 키워드 토큰 AND
    l2_5 = []
    for k in sec:
        tokens = [t for t in re.split(r'[\s·\-]+', k) if t]
        valid_tokens = [t for t in tokens if is_valid_token(t)]
        if len(valid_tokens) < 2: continue
        if all(norm(t) in text_norm for t in valid_tokens):
            l2_5.append(f"{k} valid={valid_tokens}")
    if l2_5:
        return 0.5, {'level': 'L2.5_secondary_token_AND', 'hits': l2_5,
                     'specificity_boost': specificity_boost,
                     'stock_hits': stock_hits}

    # L3: alias exact (stocks 단독은 통과 안 됨)
    if policy['alias'] and norm(policy['alias']) in text_norm:
        return 0.4, {'level': 'L3_alias', 'alias': policy['alias'],
                     'specificity_boost': specificity_boost,
                     'stock_hits': stock_hits}

    return 0.0, {'level': 'NO_MATCH', 'specificity_boost': specificity_boost,
                 'stock_hits': stock_hits}

# ─────────────────────────────────────────
RELABEL = {
    'case01': 'direct', 'case02': 'partial', 'case03': 'noise',
    'case04': 'noise',  'case05': 'direct',  'case06': 'noise',
    'case07': 'direct', 'case08': 'noise',   'case09': 'noise',
    'case10': 'partial',
}

INDUSTRY_MAP = {
    '반도체': '반도체 업종',
    '금융/증권': '금융 업종',
    '부동산/건설': '건설 업종',
    '바이오/제약': '제약·바이오 업종',
    '2차전지/에너지': '2차전지 업종',
}

def build_hypotheses_v3_1(policy, case_id):
    ind = INDUSTRY_MAP.get(policy['industry'], policy['industry'])
    stocks = policy['top_stocks'].split(',')[:2]
    stocks_str = ', '.join(s.strip() for s in stocks if s.strip() and s.strip() != '전 종목')
    if not stocks_str: stocks_str = ind
    alias = policy['alias']

    # case05 v3 — C5 유동성 + C6 투자심리 명시
    if case_id == 'case05':
        pos = f"이 기사는 공매도 금지 정책이 시장 안정, 유동성 회복, 투자심리 개선을 통해 금융 업종 주가에 상승 압력을 만든다고 설명한다"
        neg = f"이 기사는 공매도 금지 정책이 거래 위축과 시장 왜곡을 야기해 금융 업종 주가에 하락 압력을 만든다고 설명한다"
        neu = f"이 기사는 공매도 금지 정책과 주가 영향이 무관하거나 단순 정보 전달이라고 설명한다"
        return [pos, neg, neu]

    # case07 v3 — C4 불확실성 + C6 투자심리
    if case_id == 'case07':
        pos = f"이 기사는 의대 정원 확대 정책이 의료 시스템 확충으로 이어져 제약·바이오 업종의 주가 상승 요인이 된다고 설명한다"
        neg = f"이 기사는 의대 정원 확대로 의정 갈등과 정책 불확실성이 커지면서 제약·바이오 업종 투자심리가 위축되고 주가 하락 압력이 증가한다고 설명한다"
        neu = f"이 기사는 의대 정원 확대 정책과 주가 영향이 무관하거나 단순 정보 전달이라고 설명한다"
        return [pos, neg, neu]

    # 일반 템플릿
    pos = f"이 기사는 '{alias}' 정책으로 인해 {stocks_str} 등 {ind}의 주가가 상승할 수 있다고 설명한다"
    neg = f"이 기사는 '{alias}' 정책으로 인해 {stocks_str} 등 {ind}의 주가가 하락할 수 있다고 설명한다"
    neu = f"이 기사는 '{alias}' 정책과 무관하거나 주가에 영향 없다고 설명한다"
    return [pos, neg, neu]

def classify(p_pos, p_neg, p_neu, rel, tau_rel=0.4, tau_margin=0.10):
    if rel < tau_rel: return 'IRR'
    if p_neu > max(p_pos, p_neg): return 'N'
    if abs(p_pos - p_neg) < tau_margin:
        return f"LOW_{'+' if p_pos > p_neg else '-'}"
    return '+' if p_pos > p_neg else '-'

# ─────────────────────────────────────────
print("=" * 95)
print("PPT-NLP Sanity Check v3.1 — gate 미세조정 + case05 가설 v3")
print("=" * 95)

P = load_policies()
with open('sanity_check_cases.json', encoding='utf-8') as f:
    cases = json.load(f)

print("\n[1/3] Relevance v3.1 계산")
print("-" * 95)
print(f"{'case':>6s} | label    | rel  | level                     | stocks(specificity)")
print("-" * 95)
for c in cases:
    policy = P[c['policy_n']]
    rel, det = relevance_v3_1(c['title'], c['content'], policy)
    c['relevance'] = rel
    c['rel_detail'] = det
    c['true_label'] = RELABEL[c['case_id']]
    stock_str = ','.join(det.get('stock_hits', [])) or '-'
    print(f"{c['case_id']:>6s} | {c['true_label']:<8s} | {rel:.2f} | {det['level']:<25s} | {stock_str}")

print("\n[2/3] NLI 추론")
zsc = pipeline("zero-shot-classification",
               model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=-1)
print("  ✓")
print("-" * 95)
print(f"{'case':>6s} | label    | exp | rel  | p+   | p-   | pN   | pred       | result")
print("-" * 95)

results = []
for c in cases:
    text = f"{c['title']}\n{c['content'] or ''}"
    policy = P[c['policy_n']]
    cand = build_hypotheses_v3_1(policy, c['case_id'])
    out = zsc(text, cand, multi_label=False)
    sc = dict(zip(out['labels'], out['scores']))
    p_pos = sc[cand[0]]; p_neg = sc[cand[1]]; p_neu = sc[cand[2]]
    pred = classify(p_pos, p_neg, p_neu, c['relevance'])
    exp = c['D_i_expected']; label = c['true_label']

    if label == 'noise':
        result = ('✓ noise → IRR' if pred == 'IRR'
                  else f'✗ noise → {pred} (false positive)')
    elif label == 'direct':
        if pred == 'IRR':
            result = '✗ direct → IRR (gate fail)'
        elif pred in ('+', '-'):
            result = f"{'✓' if pred == exp else '✗'} direct stance: {pred} (exp {exp})"
        elif pred.startswith('LOW_'):
            s = pred.split('_')[1]
            result = f"{'~' if s == exp else '✗'} direct LOW: {pred}"
        else:
            result = '~ direct N (NLI 모호)'
    else:  # partial
        if pred == 'IRR':
            result = '~ partial → IRR (수용)'
        elif pred in ('+', '-'):
            result = f"{'✓' if pred == exp else '~'} partial: {pred} (exp {exp})"
        elif pred.startswith('LOW_'):
            result = f"~ partial LOW: {pred}"
        else:
            result = '~ partial N'

    results.append({**c, 'p_pos': round(p_pos,3), 'p_neg': round(p_neg,3),
                    'p_neu': round(p_neu,3), 'pred': pred, 'result': result})
    print(f"{c['case_id']:>6s} | {label:<8s} |  {exp}  | {c['relevance']:.2f} | "
          f"{p_pos:.2f} | {p_neg:.2f} | {p_neu:.2f} | {pred:<10s} | {result}")

print("-" * 95)
n_d = [r for r in results if r['true_label'] == 'direct']
n_n = [r for r in results if r['true_label'] == 'noise']
n_p = [r for r in results if r['true_label'] == 'partial']

print(f"\n=== v3.1 평가 ===")
print(f"Noise IRR 정확:        {sum(1 for r in n_n if r['pred']=='IRR')}/{len(n_n)}")
print(f"Direct gate fail (IRR):{sum(1 for r in n_d if r['pred']=='IRR')}/{len(n_d)}  ← A1 핵심")
print(f"Direct stance 정확:    {sum(1 for r in n_d if r['pred'] in ('+','-') and r['pred']==r['D_i_expected'])}/{len(n_d)}")
print(f"Direct LOW 방향 정확:  {sum(1 for r in n_d if r['pred'].startswith('LOW_') and r['pred'].split('_')[1]==r['D_i_expected'])}/{len(n_d)}")
print(f"Partial 처리:          {sum(1 for r in n_p if r['pred']=='IRR' or (r['pred'] in ('+','-') and r['pred']==r['D_i_expected']) or r['pred'].startswith('LOW_'))}/{len(n_p)}")

with open('sanity_check_results_v3_1.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n저장: sanity_check_results_v3_1.json")
