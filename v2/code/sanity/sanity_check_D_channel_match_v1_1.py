"""
D-2 channel_match hybrid gate sanity
====================================
질문: keyword + semantic hybrid gate가 채널을 잘 활성화하는가?
      (NLI 단독 실패 — C-1에서 확인)

설계:
  keyword_hit(a,c) = channel_keywords.json 시드 substring 매칭
  semantic_hit(a,c) = cos(emb(article), emb(channel_seed)) ≥ τ
  active(a,c) = keyword_hit OR (semantic_hit AND relevance≥0.4)
                 ※ 샘플은 이미 relevance≥0.4 통과만 사용 — semantic 조건 자동 충족

규칙 (사용자 가이드):
  - τ = 0.55 공통 (채널별 X)
  - semantic 단독 활성화 금지 (relevance 선행 필수)
  - 합격: expected 채널 중 1+ active + 무관 채널 과도 활성 X
  - 출력: binary gate + cosine score 둘 다 저장

모델: klue/roberta-base (sanity 충분, 본 분석은 large 옵션 명시)
"""
import torch, numpy as np
import json, csv, re
import openpyxl
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────
# 채널 키워드 시드 풀
# ─────────────────────────────────────────
with open('channel_keywords_v0.2.json', encoding='utf-8') as f:
    ck = json.load(f)

CHANNEL_KEYWORDS = {}
for c, info in ck['channels'].items():
    kws = []
    for grp in info['polarity_keywords'].values():
        kws.extend(grp)
    kws.extend(info['neutral_anchors'])
    CHANNEL_KEYWORDS[c] = list(set(kws))

def normalize(s):
    return re.sub(r'\s+', '', (s or '').lower())

def keyword_hit(text, channel):
    tnorm = normalize(text)
    for kw in CHANNEL_KEYWORDS[channel]:
        if kw and normalize(kw) in tnorm:
            return True, kw
    return False, None

# ─────────────────────────────────────────
# 임베딩 (KLUE-RoBERTa-base)
# ─────────────────────────────────────────
print("[1/4] 모델 로드 (klue/roberta-base, ~440MB)")
MODEL = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()
print("  ✓")

def embed(text):
    if not text or not text.strip(): text = " "
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        out = model(**enc)
    mask = enc['attention_mask'].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (summed / counts).squeeze().numpy()

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

print("[2/4] 채널 시드 임베딩 (6채널 × 시드 평균 풀링)")
channel_emb = {}
for c, kws in CHANNEL_KEYWORDS.items():
    sample = kws[:20]
    embs = np.stack([embed(kw) for kw in sample])
    channel_emb[c] = embs.mean(axis=0)
print("  ✓")

# ─────────────────────────────────────────
# 정책 카드 + relevance
# ─────────────────────────────────────────
with open('policy_cards_v0.2.csv', encoding='utf-8-sig') as f:
    cards = {int(c['pn']): c for c in csv.DictReader(f)}
wb = openpyxl.load_workbook('50_정책마스터.xlsx', read_only=True, data_only=True)
ws = wb['1_50정책_마스터']
KW = {}
for r in list(ws.iter_rows(values_only=True))[3:]:
    if not r or not r[0]: continue
    KW[int(r[0])] = {'kw1': r[10] or '', 'kw2': r[11] or '', 'not_words': r[13] or ''}
wb.close()

def parse_or(s):
    if not s or str(s).lower() == 'none': return []
    return [t.strip() for t in re.split(r'\s+OR\s+', str(s).strip())]
def is_valid_token(t): return len(t) >= 2 and not t.isdigit()
def relevance_v3_1(title, content, kw):
    raw = (title or '') + ' ' + (content or '')
    t = normalize(raw)
    for nw in parse_or(kw['not_words']):
        if nw and normalize(nw) in t: return 0.0
    pri = parse_or(kw['kw1']); sec = parse_or(kw['kw2'])
    for k in pri:
        if k and normalize(k) in t: return 1.0
    for k in sec:
        if k and normalize(k) in t: return 0.7
    for k in pri:
        toks = [x for x in re.split(r'[\s·\-]+', k) if x and is_valid_token(x)]
        if len(toks) >= 2 and all(normalize(x) in t for x in toks): return 0.7
    for k in sec:
        toks = [x for x in re.split(r'[\s·\-]+', k) if x and is_valid_token(x)]
        if len(toks) >= 2 and all(normalize(x) in t for x in toks): return 0.5
    return 0.0

# ─────────────────────────────────────────
# 기사 샘플 — C-1과 동일 4정책 × 6건
# ─────────────────────────────────────────
TARGET = [3, 8, 15, 28]
EXPECTED = {
    3:  ['C3', 'C4'],
    8:  ['C1', 'C5'],   # K칩스법: C3·C6도 OK라고 사용자 언급, 핵심은 C1·C5
    15: ['C5', 'C6'],
    28: ['C2', 'C4'],
}
EXPECTED_BONUS = {  # 사용자 표현 "K칩스법은 C1도 맞고 C3도 C6도" — bonus 허용
    8: ['C3', 'C6'],
}

print("[3/4] 기사 샘플링")
pool = defaultdict(list)
with open('50_전체기사_99539건.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        try: pn = int(row['policy_n'])
        except: continue
        if pn not in TARGET: continue
        if len(pool[pn]) >= 30: continue
        try: rd = int(row['rel_day'])
        except: continue
        if -7 <= rd <= 5: pool[pn].append(row)

selected = {}
for pn in TARGET:
    rated = []
    for r in pool[pn]:
        rel = relevance_v3_1(r['title'], r['content'], KW[pn])
        if rel >= 0.4:
            rated.append((rel, abs(int(r['rel_day'])), r))
    rated.sort(key=lambda x: (-x[0], x[1]))
    selected[pn] = [t[2] for t in rated[:6]]
    print(f"  pn={pn}: {len(rated)} 후보 → 상위 {len(selected[pn])} 선별")

TAU = 0.55
print(f"\n[4/4] Hybrid gate 평가 (τ={TAU})")
print("=" * 100)

results = []
for pn in TARGET:
    card = cards[pn]
    exp = set(EXPECTED[pn])
    bonus = set(EXPECTED_BONUS.get(pn, []))
    print(f"\n--- pn={pn} {card['alias']} ({card['type_subtype']}, D={card['D_i_initial']})")
    print(f"    expected: {sorted(exp)}, bonus_ok: {sorted(bonus) if bonus else '-'}")

    for art in selected[pn]:
        text = f"{art['title']} {art['content'] or ''}"
        art_emb = embed(text)

        ch_res = {}
        for c in ['C1','C2','C3','C4','C5','C6']:
            kw_h, kw_w = keyword_hit(text, c)
            cos = cos_sim(art_emb, channel_emb[c])
            sem_h = cos >= TAU
            active = kw_h or sem_h  # relevance≥0.4 이미 보장
            ch_res[c] = {'kw': kw_h, 'kw_word': kw_w or '', 'cos': round(cos,3),
                         'sem': sem_h, 'active': active}

        active_set = {c for c, r in ch_res.items() if r['active']}
        overlap = active_set & exp
        bonus_match = active_set & bonus
        false_active = active_set - exp - bonus

        # 판정
        if overlap and len(false_active) <= 1:
            verdict = '✓ good' if overlap == exp else '✓ partial-OK'
        elif overlap:
            verdict = '~ partial-over'
        elif active_set & bonus:
            verdict = '~ bonus only'
        else:
            verdict = '✗ no overlap'

        rd = int(art['rel_day'])
        # cos vector 표시
        cos_str = ' '.join(f"{c}={ch_res[c]['cos']:.2f}{'●' if ch_res[c]['active'] else ' '}"
                           for c in ['C1','C2','C3','C4','C5','C6'])
        active_str = ','.join(sorted(active_set)) or '-'
        kw_str = ','.join(c for c in ch_res if ch_res[c]['kw']) or '-'
        print(f"\n  [D{rd:+d}] \"{art['title'][:50]}\"")
        print(f"        {cos_str}")
        print(f"        active={active_str}  kw_hit={kw_str}  overlap={sorted(overlap)}  false={sorted(false_active)}  → {verdict}")

        results.append({
            'pn': pn, 'title': art['title'][:60], 'rel_day': rd,
            **{f'{c}_cos': ch_res[c]['cos'] for c in ['C1','C2','C3','C4','C5','C6']},
            **{f'{c}_kw': ch_res[c]['kw'] for c in ['C1','C2','C3','C4','C5','C6']},
            **{f'{c}_active': ch_res[c]['active'] for c in ['C1','C2','C3','C4','C5','C6']},
            'active_set': sorted(active_set), 'overlap': sorted(overlap),
            'false_active': sorted(false_active), 'verdict': verdict,
        })

print("\n" + "=" * 100)
print("\n=== 정책별 평균 cos + 활성률 ===")
print(f"{'pn':>3s} {'정책':<18s} {'expected':<10s} | avg cos (C1-C6)                       | 활성률 C1-C6")
print("-" * 100)
for pn in TARGET:
    rs = [r for r in results if r['pn'] == pn]
    avg_cos = {c: np.mean([r[f'{c}_cos'] for r in rs]) for c in ['C1','C2','C3','C4','C5','C6']}
    rate = {c: sum(1 for r in rs if r[f'{c}_active']) / len(rs) for c in ['C1','C2','C3','C4','C5','C6']}
    cos_s = ' '.join(f"{avg_cos[c]:.2f}" for c in ['C1','C2','C3','C4','C5','C6'])
    rate_s = ' '.join(f"{rate[c]:.2f}" for c in ['C1','C2','C3','C4','C5','C6'])
    exp_s = '/'.join(EXPECTED[pn])
    print(f"{pn:>3d} {cards[pn]['alias']:<18s} {exp_s:<10s} | {cos_s} | {rate_s}")

# 합격 판정
print("\n=== Sanity 판정 (사용자 합격 기준) ===")
print("  기준: expected 채널 1+ active AND false_active ≤ 명백히 무관 채널 다수 X")
for pn in TARGET:
    rs = [r for r in results if r['pn'] == pn]
    n_overlap = sum(1 for r in rs if r['overlap'])
    n_partial = sum(1 for r in rs if r['verdict'] in ('✓ good', '✓ partial-OK', '~ partial-over'))
    n_false = sum(len(r['false_active']) for r in rs) / len(rs)
    verdict_pn = '✓ PASS' if n_overlap >= len(rs) * 0.5 and n_false <= 2 else '⚠ MIXED' if n_overlap >= 1 else '✗ FAIL'
    print(f"  pn={pn} {cards[pn]['alias']:<18s}: expected 1+ active {n_overlap}/{len(rs)}, "
          f"평균 false_active {n_false:.1f} → {verdict_pn}")

with open('sanity_D_channel_match_v1_1_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n저장: sanity_D_channel_match_v1_1_results.json")
