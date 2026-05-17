"""
Cohen's κ 측정 — 라벨러 2명 독립 라벨 비교
=========================================
입력: splits/pilot_30_labels_labelerA.csv, _labelerB.csv
출력: 콘솔 + splits/pilot_30_disagreement.csv

기준: κ ≥ 0.7 (relevance, stance_label 둘 다)
미달 시 §5.3 절차로 가이드 v2 보정

실행 전 sklearn 필요:
  .venv-nlp/bin/pip install scikit-learn pandas
"""
import csv, sys, os

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("ERROR: scikit-learn 미설치. .venv-nlp/bin/pip install scikit-learn")
    sys.exit(1)

PATH_A = 'splits/pilot_30_labels_ai.csv'        # 사용자 라벨 (파일명만 _ai, 내용은 user)
PATH_B = 'splits/pilot_30_labels_claude.csv'    # Claude 라벨
LABEL_A_NAME = 'User'
LABEL_B_NAME = 'Claude'

if not (os.path.exists(PATH_A) and os.path.exists(PATH_B)):
    print(f"라벨 파일 미발견:")
    print(f"  {PATH_A}: {'OK' if os.path.exists(PATH_A) else '없음'}")
    print(f"  {PATH_B}: {'OK' if os.path.exists(PATH_B) else '없음'}")
    print(f"\n라벨링 완료 후 실행하세요.")
    sys.exit(0)

def load(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))

a = load(PATH_A)
b = load(PATH_B)

# news_id 또는 (pn, title)로 매칭
def key(r): return (str(r.get('pn','')), r.get('title','')[:30])
a_map = {key(r): r for r in a}
b_map = {key(r): r for r in b}
common = sorted(set(a_map.keys()) & set(b_map.keys()))
print(f"공통 케이스: {len(common)}건 ({LABEL_A_NAME}: {len(a)}, {LABEL_B_NAME}: {len(b)})")

def parse_rel(v):
    try: return int(v)
    except: return -1

a_rel = [parse_rel(a_map[k]['relevance']) for k in common]
b_rel = [parse_rel(b_map[k]['relevance']) for k in common]
a_st = [a_map[k]['stance_label'].strip() or 'neutral' for k in common]
b_st = [b_map[k]['stance_label'].strip() or 'neutral' for k in common]

kappa_rel = cohen_kappa_score(a_rel, b_rel)
kappa_st = cohen_kappa_score(a_st, b_st)

print(f"\n=== Cohen's κ ===")
print(f"relevance:    κ = {kappa_rel:.3f}   {'✓ PASS' if kappa_rel >= 0.7 else '✗ FAIL — 가이드 v2 필요'}")
print(f"stance_label: κ = {kappa_st:.3f}   {'✓ PASS' if kappa_st >= 0.7 else '✗ FAIL — 가이드 v2 필요'}")

# 불일치 매트릭스
from collections import Counter
rel_pairs = Counter(zip(a_rel, b_rel))
st_pairs = Counter(zip(a_st, b_st))
print(f"\n=== relevance 페어 분포 (User→Claude) ===")
for (av, bv), n in sorted(rel_pairs.items()):
    mark = ' ' if av == bv else '✗'
    print(f"  {mark} User={av} Claude={bv}: {n}")
print(f"\n=== stance_label 페어 분포 (User→Claude) ===")
for (av, bv), n in sorted(st_pairs.items()):
    mark = ' ' if av == bv else '✗'
    print(f"  {mark} User={av:<10s} Claude={bv:<10s}: {n}")

# 불일치 저장
dis_rows = []
for k in common:
    if a_map[k]['relevance'] != b_map[k]['relevance'] or \
       (a_map[k]['stance_label'].strip() or 'neutral') != (b_map[k]['stance_label'].strip() or 'neutral'):
        dis_rows.append({
            'pn': a_map[k].get('pn',''),
            'title': a_map[k].get('title','')[:60],
            'User_relevance': a_map[k]['relevance'],
            'Claude_relevance': b_map[k]['relevance'],
            'User_stance': a_map[k]['stance_label'],
            'Claude_stance': b_map[k]['stance_label'],
            'User_reason': a_map[k].get('reason',''),
            'Claude_reason': b_map[k].get('reason',''),
        })

if dis_rows:
    keys = list(dis_rows[0].keys())
    with open('splits/pilot_30_disagreement.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(dis_rows)
    print(f"\n불일치 {len(dis_rows)}건 저장: splits/pilot_30_disagreement.csv")
    print("→ 가이드 v2 보정 시 이 불일치 케이스들을 §3 어려운 케이스에 추가")
else:
    print(f"\n불일치 없음 (완전 일치)")
