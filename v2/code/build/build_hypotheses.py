"""
가설 템플릿 자동 생성 — 8 subtype × (support/contradict/neutral)
================================================================
v2.6 §12.14 매트릭스 동결. case별 가설 작성 금지.

입력: policy_cards_v0.1.csv
출력:
  policy_cards_v0.2.csv (review_status·review_note 추가, #11 D_crisis_shock 정정)
  policy_hypotheses_v0.1.csv (50건 × 3가설)

상태:
  v0.x = 템플릿 개발용 초안 (분석 동결본 X)
  검토 필요 6건: pn=11 (CAR sign), pn=26·38·39·40 (master type), pn=29 (정책 정체)
"""
import csv

# ─────────────────────────────────────────
# v2.6 §12.14 가설 템플릿 동결
# (subtype) → {D_i_default, supports_pos, supports_neg}
# 'supports_pos' = D_i=+1일 때의 support 가설 (주가 상승 방향)
# 'supports_neg' = D_i=-1일 때의 support 가설 (주가 하락 방향)
# ─────────────────────────────────────────
SUBTYPE_TEMPLATES = {
    'A_tax_support': {
        'default_D_i': '+',
        'channels': ['C1', 'C5'],
        'pos': "'{policy}' 정책이 세부담 감소와 자금 여력 개선을 통해 {stocks} 등 {industry}의 주가에 상승 압력을 만든다",
        'neg': "'{policy}' 정책이 세부담 증가나 정책 효과 부진으로 {stocks} 등 {industry}의 주가에 하락 압력을 만든다",
    },
    'B_restriction_negative': {
        'default_D_i': '-',
        'channels': ['C2', 'C4'],
        'pos': "'{policy}' 정책이 시장 정상화와 신뢰 회복을 통해 {stocks} 등 {industry}의 주가에 상승 압력을 만든다",
        'neg': "'{policy}' 정책이 매출 위축과 정책 불확실성 확대로 {stocks} 등 {industry}의 주가에 하락 압력을 만든다",
    },
    'B_market_stabilizer': {
        'default_D_i': '+',
        'channels': ['C5', 'C6'],
        'pos': "'{policy}' 정책이 시장 안정·유동성 회복·투자심리 개선을 통해 {stocks} 등 {industry}의 주가에 상승 압력을 만든다",
        'neg': "'{policy}' 정책이 거래 위축과 시장 왜곡으로 {stocks} 등 {industry}의 주가에 하락 압력을 만든다",
    },
    'C_industrial_support': {
        'default_D_i': '+',
        'channels': ['C3', 'C6'],
        'pos': "'{policy}' 정책이 설비투자 확대·인프라 구축·시장 신뢰 강화를 통해 {stocks} 등 {industry}의 주가에 상승 압력을 만든다",
        'neg': "'{policy}' 정책이 실행 지연과 정책 효과 부진으로 {stocks} 등 {industry}의 주가에 하락 압력을 만든다",
    },
    'D_crisis_relief': {
        'default_D_i': '+',
        'channels': ['C4', 'C5', 'C6'],
        'pos': "'{policy}' 정책이 유동성 공급·신뢰 회복·불확실성 완화를 통해 {stocks} 등 {industry}의 주가에 상승 압력을 만든다",
        'neg': "'{policy}' 정책이 정책 효과 미흡과 추가 위기 우려 확산으로 {stocks} 등 {industry}의 주가에 하락 압력을 만든다",
    },
    'D_crisis_shock': {
        'default_D_i': '-',
        'channels': ['C4', 'C6'],
        'pos': "'{policy}' 충격이 시장 정화와 점진적 신뢰 회복으로 {stocks} 등 {industry}의 주가에 상승 압력을 만든다",
        'neg': "'{policy}' 충격이 패닉 확산·신뢰 훼손·정책 불확실성 증대로 {stocks} 등 {industry}의 주가에 하락 압력을 만든다",
    },
    'E_foreign_benefit': {
        'default_D_i': '+',
        'channels': ['C2', 'C3'],
        'pos': "'{policy}'이(가) 한국 기업의 수출 수혜와 공급망 우위를 통해 {stocks} 등 {industry}의 주가에 상승 압력을 만든다",
        'neg': "'{policy}'이(가) 글로벌 경쟁 격화와 기대 미달로 {stocks} 등 {industry}의 주가에 하락 압력을 만든다",
    },
    'E_foreign_shock': {
        'default_D_i': '-',
        'channels': ['C2', 'C3', 'C4'],
        'pos': "'{policy}' 충격에 대해 한국 기업의 대안 시장 확보와 국산화 가속으로 {stocks} 등 {industry}의 주가에 상승 압력을 만든다",
        'neg': "'{policy}' 충격이 수출 충격·공급망 위축·정책 불확실성 확대로 {stocks} 등 {industry}의 주가에 하락 압력을 만든다",
    },
}

INDUSTRY_MAP = {
    '반도체': '반도체 업종',
    '금융/증권': '금융 업종',
    '부동산/건설': '건설 업종',
    '바이오/제약': '제약·바이오 업종',
    '2차전지/에너지': '2차전지 업종',
}

# ─────────────────────────────────────────
# build_hypothesis(card) — 핵심 함수
# ─────────────────────────────────────────
def build_hypothesis(card, top_stocks=''):
    """
    카드 기준 NLI 가설 3개 자동 생성:
      support     : D_i 방향 지지 가설
      contradict  : D_i 반대 방향 가설
      neutral     : 무관 가설
    """
    subtype = card['type_subtype']
    if subtype not in SUBTYPE_TEMPLATES:
        return None, None, None
    tpl = SUBTYPE_TEMPLATES[subtype]
    D_i = card['D_i_initial']
    if D_i not in ('+', '-'):
        D_i = tpl['default_D_i']

    industry = INDUSTRY_MAP.get(card['industry'], card['industry'])
    stocks = ', '.join([s.strip() for s in top_stocks.split(',')[:2]
                        if s.strip() and s.strip() != '전 종목']) or industry
    policy = card['alias']

    pos = tpl['pos'].format(policy=policy, stocks=stocks, industry=industry)
    neg = tpl['neg'].format(policy=policy, stocks=stocks, industry=industry)
    neutral = f"이 기사는 '{policy}' 정책과 무관하거나 정책의 주가 영향이 아닌 단순 사실 전달이다"

    if D_i == '+':
        return pos, neg, neutral
    else:  # D_i == '-'
        return neg, pos, neutral

# ─────────────────────────────────────────
# v0.2: review_status·review_note 추가 + #11 정정
# ─────────────────────────────────────────
REVIEW_MARKS = {
    11: ('needs_review',
         'D_crisis_shock 재분류 (정부 정리 발표가 시장 충격으로 가격 반영). CAR sign 확정 시 동결.'),
    26: ('needs_review',
         'master_type 시트=C, 문서 본문=A(취득세·양도세 감면). 시트 오류 가능.'),
    29: ('conflict',
         '정책 정체 불명. 정책마스터 약칭=9·13 대책, 문서 본문=윤석열 규제완화 패키지(2022.6.21). 발표일·D_i 재확인 필수.'),
    38: ('needs_review',
         'master_type 시트=C, 문서 본문=D(코로나 백신·CMO 지원). 시트 오류 가능.'),
    39: ('needs_review',
         'master_type 시트=D, 문서 본문=B(의대정원 강행·의료대란). 시트 오류 가능.'),
    40: ('needs_review',
         'master_type 시트=E, 문서 본문=A(CDMO 특별법·세제 혁신). 시트 오류 가능.'),
}

# 5개 은행 퇴출(pn=11) subtype 재분류
SUBTYPE_OVERRIDE = {
    11: 'D_crisis_shock',
}

# ─────────────────────────────────────────
# 처리
# ─────────────────────────────────────────
with open('policy_cards_v0.1.csv', encoding='utf-8-sig') as f:
    cards = list(csv.DictReader(f))

# 정책 종목 매핑 (정책마스터에서 추출했던 정보 유지)
import openpyxl
wb = openpyxl.load_workbook('50_정책마스터.xlsx', read_only=True, data_only=True)
ws = wb['1_50정책_마스터']
rows = list(ws.iter_rows(values_only=True))
wb.close()
TOP_STOCKS = {}
for r in rows[3:]:
    if not r or not r[0]: continue
    TOP_STOCKS[int(r[0])] = r[9] or ''

# v0.2 카드 + 가설 동시 생성
cards_v02 = []
hyps_rows = []
for card in cards:
    pn = int(card['pn'])
    # subtype override
    if pn in SUBTYPE_OVERRIDE:
        new_sub = SUBTYPE_OVERRIDE[pn]
        card['type_subtype'] = new_sub
        # 채널 재설정
        for c in ['C1','C2','C3','C4','C5','C6']: card[c] = ''
        tpl = SUBTYPE_TEMPLATES[new_sub]
        for c in tpl['channels']:
            card[c] = 'ON'

    # review 마크
    st, note = REVIEW_MARKS.get(pn, ('ok', ''))
    card['review_status'] = st
    card['review_note'] = note
    card['status'] = 'v0.2_dev'
    cards_v02.append(card)

    # 가설 생성
    top = TOP_STOCKS.get(pn, '')
    sup, con, neu = build_hypothesis(card, top)
    hyps_rows.append({
        'pn': pn,
        'alias': card['alias'],
        'industry': card['industry'],
        'type_subtype': card['type_subtype'],
        'D_i_initial': card['D_i_initial'],
        'active_channels': ','.join([c for c in ['C1','C2','C3','C4','C5','C6'] if card.get(c)=='ON']),
        'hypothesis_support': sup,
        'hypothesis_contradict': con,
        'hypothesis_neutral': neu,
        'review_status': st,
    })

# v0.2 저장 (review 컬럼이 헤더 끝에 오도록)
keys_order = ['pn','industry','alias','date','master_type','type_subtype','type_match',
              'D_i_initial','k_i','C1','C2','C3','C4','C5','C6','channel_signs',
              'rationale','status','review_status','review_note']
with open('policy_cards_v0.2.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys_order)
    w.writeheader()
    for c in cards_v02:
        row = {k: c.get(k, '') for k in keys_order}
        w.writerow(row)

# 가설 저장
with open('policy_hypotheses_v0.1.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(hyps_rows[0].keys()))
    w.writeheader()
    w.writerows(hyps_rows)

# ─────────────────────────────────────────
# 콘솔 보고
# ─────────────────────────────────────────
print(f"v0.2 카드 {len(cards_v02)}건 + 가설 {len(hyps_rows)}건 생성")
print()
print("=== review_status 분포 ===")
from collections import Counter
rev_count = Counter(c['review_status'] for c in cards_v02)
for k, v in rev_count.items():
    print(f"  {k}: {v}")
print()
print("=== 검토 필요 6건 ===")
for c in cards_v02:
    if c['review_status'] != 'ok':
        print(f"  #{c['pn']:>2s} {c['alias']:<18s} [{c['review_status']}]")
        print(f"      {c['review_note']}")
print()

# 가설 샘플 — 8 subtype × 1건씩
print("=== 가설 샘플 (8 subtype 각 1건) ===")
seen = set()
for h in hyps_rows:
    if h['type_subtype'] in seen: continue
    seen.add(h['type_subtype'])
    print(f"\n[#{h['pn']} {h['alias']}] {h['type_subtype']} (D_i={h['D_i_initial']})")
    print(f"  active channels: {h['active_channels']}")
    print(f"  SUP: {h['hypothesis_support'][:100]}")
    print(f"  CON: {h['hypothesis_contradict'][:100]}")
    if len(seen) == 8: break

print("\n저장:")
print("  policy_cards_v0.2.csv (review 컬럼 + #11 정정)")
print("  policy_hypotheses_v0.1.csv (50건 × 3가설)")
