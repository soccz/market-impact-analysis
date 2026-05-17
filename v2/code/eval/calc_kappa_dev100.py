"""
Dev 100 사람-사람 Cohen's κ 측정
================================
입력: splits/dev_labels_labelerA.csv, dev_labels_labelerB.csv
출력: 콘솔 + splits/dev_disagreement.csv

기준: κ ≥ 0.7 (relevance, stance_label 둘 다)
미달 시 §5.3 절차 (불일치 유형화 → 가이드 v2)
"""
import csv, sys, os
try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("ERROR: scikit-learn 미설치")
    sys.exit(1)

PATH_A = 'splits/dev_labels_labelerA.csv'
PATH_B = 'splits/dev_labels_labelerB.csv'

def load(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))

a = load(PATH_A); b = load(PATH_B)

# news_id 기반 매칭 (Dev는 같은 pn에 여러 기사)
def key(r):
    nid = r.get('news_id', '').strip()
    if nid: return nid
    return (str(r.get('pn','')), r.get('title','')[:30])

a_map = {key(r): r for r in a}
b_map = {key(r): r for r in b}
common = sorted(set(a_map.keys()) & set(b_map.keys()))
print(f"공통 케이스: {len(common)}건 (A: {len(a)}, B: {len(b)})")

if len(common) < 50:
    print(f"⚠ 공통 케이스 부족. A·B 파일 매칭 확인 필요.")
    sys.exit(1)

def parse_rel(v):
    try: return int(v)
    except: return -1

a_rel = [parse_rel(a_map[k]['relevance']) for k in common]
b_rel = [parse_rel(b_map[k]['relevance']) for k in common]
a_st = [a_map[k]['stance_label'].strip() or 'neutral' for k in common]
b_st = [b_map[k]['stance_label'].strip() or 'neutral' for k in common]

kappa_rel = cohen_kappa_score(a_rel, b_rel)
kappa_st = cohen_kappa_score(a_st, b_st)

# Subset κ — relevance>0인 케이스에서만 stance κ
mask = [(ar > 0 and br > 0) for ar, br in zip(a_rel, b_rel)]
a_st_sub = [s for s, m in zip(a_st, mask) if m]
b_st_sub = [s for s, m in zip(b_st, mask) if m]
kappa_st_relevant = cohen_kappa_score(a_st_sub, b_st_sub) if len(a_st_sub) >= 10 else None

print(f"\n=== Dev 100 사람-사람 Cohen's κ ===")
print(f"relevance:                  κ = {kappa_rel:.3f}   {'✓ PASS' if kappa_rel >= 0.7 else '✗ FAIL'}")
print(f"stance_label (전체):         κ = {kappa_st:.3f}   {'✓ PASS' if kappa_st >= 0.7 else '✗ FAIL'}")
if kappa_st_relevant is not None:
    print(f"stance_label (rel>0 only):  κ = {kappa_st_relevant:.3f}  ({len(a_st_sub)}건, relevance=0 제외)")

# 페어 분포
from collections import Counter
rel_pairs = Counter(zip(a_rel, b_rel))
st_pairs = Counter(zip(a_st, b_st))
print(f"\n=== relevance 페어 분포 (A→B) ===")
for (av, bv), n in sorted(rel_pairs.items()):
    mark = ' ' if av == bv else '✗'
    print(f"  {mark} A={av} B={bv}: {n}")
print(f"\n=== stance_label 페어 분포 (A→B) ===")
for (av, bv), n in sorted(st_pairs.items()):
    mark = ' ' if av == bv else '✗'
    print(f"  {mark} A={av:<10s} B={bv:<10s}: {n}")

# 불일치 추출
dis_rows = []
for k in common:
    a_r, b_r = a_map[k]['relevance'], b_map[k]['relevance']
    a_s = (a_map[k]['stance_label'].strip() or 'neutral')
    b_s = (b_map[k]['stance_label'].strip() or 'neutral')
    if a_r != b_r or a_s != b_s:
        dis_rows.append({
            'news_id': a_map[k].get('news_id', ''),
            'pn': a_map[k].get('pn',''),
            'alias': a_map[k].get('alias',''),
            'title': a_map[k].get('title','')[:60],
            'A_relevance': a_r, 'B_relevance': b_r,
            'A_stance': a_s, 'B_stance': b_s,
            'A_reason': a_map[k].get('reason','')[:80],
            'B_reason': b_map[k].get('reason','')[:80],
        })

print(f"\n불일치 {len(dis_rows)}/{len(common)}건")
if dis_rows:
    with open('splits/dev_disagreement.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dis_rows[0].keys()))
        w.writeheader()
        w.writerows(dis_rows)
    print(f"저장: splits/dev_disagreement.csv")

# 불일치 유형 분류 (rel 차이 / stance 차이 / 둘 다)
rel_only = sum(1 for d in dis_rows if d['A_relevance'] != d['B_relevance'] and d['A_stance'] == d['B_stance'])
st_only = sum(1 for d in dis_rows if d['A_relevance'] == d['B_relevance'] and d['A_stance'] != d['B_stance'])
both = sum(1 for d in dis_rows if d['A_relevance'] != d['B_relevance'] and d['A_stance'] != d['B_stance'])
print(f"\n  relevance만 차이: {rel_only}")
print(f"  stance만 차이:    {st_only}")
print(f"  둘 다 차이:       {both}")

# 정책별 불일치 빈도
pn_dis = Counter(d['pn'] for d in dis_rows)
print(f"\n=== 불일치 빈도 top 5 정책 ===")
for pn, n in pn_dis.most_common(5):
    print(f"  pn={pn}: {n}건")
