"""
Figure F6~F10 생성 — 정책카드 매트릭스·sanity 표·κ 비교·라벨 분포
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import csv
from collections import Counter, defaultdict

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

C = {
    'bg': '#1a1d24', 'panel': '#252932', 'text': '#e6e9ef', 'subtext': '#8a93a6',
    'accent1': '#5fbf8f', 'accent2': '#e88c5a', 'accent3': '#7a9cf2',
    'accent4': '#d96e85', 'accent5': '#b89adb', 'grid': '#363b46',
}

def setup_dark(ax, title=None):
    ax.set_facecolor(C['bg'])
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title, color=C['text'], fontsize=14, fontweight='bold', pad=15)

# ========================================
# F6 — 8 type_subtype × 50건 정책 매트릭스 (산업 × subtype heatmap)
# ========================================
with open('policy_cards_v1.1.csv', encoding='utf-8-sig') as f:
    cards = list(csv.DictReader(f))

# 산업 순서·subtype 순서
ind_order = ['반도체', '금융/증권', '부동산/건설', '바이오/제약', '2차전지/에너지']
sub_order = ['A_tax_support', 'B_restriction_negative', 'B_market_stabilizer',
             'C_industrial_support', 'D_crisis_relief', 'D_crisis_shock',
             'E_foreign_benefit', 'E_foreign_shock']
sub_short = ['A_tax', 'B_reg-', 'B_stab+', 'C_indu', 'D_relief', 'D_shock', 'E_ben+', 'E_shock-']

mat = np.zeros((len(ind_order), len(sub_order)))
for c in cards:
    if c['industry'] in ind_order and c['type_subtype'] in sub_order:
        i = ind_order.index(c['industry'])
        j = sub_order.index(c['type_subtype'])
        mat[i, j] += 1

fig, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
setup_dark(ax, "Figure 6 · 8 type_subtype × 5산업 정책 분포 (50건)")

im = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=mat.max())
ax.set_xticks(range(len(sub_order)))
ax.set_xticklabels(sub_short, rotation=30, ha='right', color=C['text'])
ax.set_yticks(range(len(ind_order)))
ax.set_yticklabels(ind_order, color=C['text'])
ax.tick_params(axis='both', colors=C['subtext'])

# 셀에 숫자 표시
for i in range(len(ind_order)):
    for j in range(len(sub_order)):
        v = int(mat[i, j])
        if v > 0:
            ax.text(j, i, v, ha='center', va='center',
                    color='black' if v >= 2 else C['text'], fontsize=12, fontweight='bold')

# 합계
totals_sub = mat.sum(axis=0).astype(int)
totals_ind = mat.sum(axis=1).astype(int)
for j, t in enumerate(totals_sub):
    ax.text(j, -0.7, f'Σ {t}', ha='center', color=C['accent2'], fontsize=10, fontweight='bold')
for i, t in enumerate(totals_ind):
    ax.text(len(sub_order)-0.3, i, f'  Σ {t}', va='center', color=C['accent2'], fontsize=10, fontweight='bold')

ax.set_xlabel('Subtype (8개)', color=C['text'], fontsize=11, labelpad=15)
ax.set_ylabel('산업 (5개)', color=C['text'], fontsize=11)

plt.tight_layout()
plt.savefig('figures/F6_subtype_matrix.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F6_subtype_matrix.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F6")

# ========================================
# F7 — Sanity 시리즈 4종 결과 통합 표
# ========================================
fig, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
setup_dark(ax, "Figure 7 · Sanity 시리즈 4종 결과 — 실패에서 얻은 결정들")
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

rows = [
    ('v3.1 relevance gate', '10건 케이스 분류',
     'noise IRR 4/4 ✓\ndirect 통과 3/3 ✓', 'BigKinds noise 정확 분리', C['accent1']),
    ('C-1 채널 NLI', '4정책 × 6채널 NLI',
     '4/4 expected 불일치 ✗\n점수 0.2~0.36 (변별 X)', '"NLI는 채널 못 잡는다"\n→ 책임 분리 결정', C['accent4']),
    ('D-2 hybrid gate', '4정책 × keyword+임베딩',
     '3/4 PASS ✓\n공매도금지 C5·C6 활성 ✓', '표면감성 우회 — 핵심 차별점\n증명 사례', C['accent1']),
    ('v1.1 D-4·D-5', 'C4 시드 정제 + subtype 매핑',
     'K칩스법 C4 0.83→0.50\n공매도 C4 1.00 잔존', '키워드 정제 한계\n→ fine-tune 필요', C['accent2']),
]

# 헤더
headers = ['Sanity', '대상', '결과', '도출 결정']
col_x = [0.5, 4.0, 7.0, 10.5]
col_w = [3.3, 2.8, 3.3, 3.3]
for h, x, w in zip(headers, col_x, col_w):
    box = FancyBboxPatch((x, 5.6), w, 0.6, boxstyle="round,pad=0.05,rounding_size=0.1",
                          facecolor=C['panel'], edgecolor=C['accent3'], linewidth=1.5)
    ax.add_patch(box)
    ax.text(x+w/2, 5.9, h, ha='center', va='center', color=C['accent3'], fontsize=11, fontweight='bold')

# 행
for ri, (sanity, target, result, decision, color) in enumerate(rows):
    y = 4.6 - ri*1.2
    cells = [sanity, target, result, decision]
    for ci, (val, x, w) in enumerate(zip(cells, col_x, col_w)):
        box = FancyBboxPatch((x, y-0.4), w, 0.95, boxstyle="round,pad=0.05,rounding_size=0.1",
                              facecolor=C['panel'], edgecolor=color if ci == 0 else C['grid'], linewidth=1)
        ax.add_patch(box)
        if ci == 0: text_c = color; fw = 'bold'
        elif ci == 2: text_c = color; fw = 'normal'
        elif ci == 3: text_c = C['accent5']; fw = 'bold'
        else: text_c = C['text']; fw = 'normal'
        ax.text(x+w/2, y+0.05, val, ha='center', va='center', color=text_c,
                fontsize=9, fontweight=fw)

plt.tight_layout()
plt.savefig('figures/F7_sanity_series.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F7_sanity_series.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F7")

# ========================================
# F8 — 공매도 금지 — C-1 vs D-2 비교
# ========================================
fig, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
setup_dark(ax, "Figure 8 · 공매도 금지 3차 (B_market_stabilizer, D=+) — C-1 NLI vs D-2 hybrid gate 활성률")

# 공매도금지 C-1 결과 (NLI multi-label): 4/6 dominant C2 (잘못)
# D-2 결과: C5 100% · C6 67% (정확)
channels = ['C1\n비용', 'C2\n수요', 'C3\n공급', 'C4\n규제', 'C5\n유동성', 'C6\n심리']
c1_rate = [0.17, 0.67, 0.00, 0.17, 0.17, 0.17]   # NLI dominant 빈도
d2_rate = [0.17, 0.17, 0.00, 1.00, 1.00, 0.67]   # hybrid gate 활성률

x = np.arange(len(channels))
w = 0.35
ax.bar(x - w/2, c1_rate, w, label='C-1 (NLI dominant)', color=C['accent4'], edgecolor=C['accent4'])
ax.bar(x + w/2, d2_rate, w, label='D-2 (hybrid gate 활성)', color=C['accent1'], edgecolor=C['accent1'])

# expected 표시
exp = [False, False, False, False, True, True]
for i, e in enumerate(exp):
    if e:
        ax.axvspan(i-0.5, i+0.5, alpha=0.08, color=C['accent1'])
        ax.text(i, 1.08, '★ expected', ha='center', color=C['accent1'], fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(channels, color=C['text'], fontsize=10)
ax.set_ylabel('활성률 / dominant 빈도', color=C['text'], fontsize=11)
ax.tick_params(axis='y', colors=C['subtext'])
ax.set_ylim(0, 1.15)
ax.legend(loc='upper left', facecolor=C['panel'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=10)
ax.grid(axis='y', color=C['grid'], linestyle='--', alpha=0.3)

ax.text(2.5, -0.18,
        'C-1 NLI: 표면감성에 끌려 C2(거래확대) dominant — 잘못된 분류 │ D-2 gate: C5·C6 활성 — 시장 안정·심리 정확',
        ha='center', color=C['accent2'], fontsize=10, style='italic', transform=ax.transData)

plt.tight_layout()
plt.savefig('figures/F8_shortsell_comparison.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F8_shortsell_comparison.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F8")

# ========================================
# F9 — κ 비교 막대 (pilot 30 / Dev 100 / Zero-shot)
# ========================================
fig, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
setup_dark(ax, "Figure 9 · Cohen's κ 비교 — 라벨러 합의도와 모델 격차")

groups = ['Pilot 30\n(사람-AI)', 'Dev 100\n(사람-사람)', 'Zero-shot\n(사람-모델)']
relevance_kappa = [0.946, 0.968, 0.386]
stance_kappa = [0.938, 0.946, 0.325]
stance_rel_kappa = [None, 0.920, 0.177]

x = np.arange(len(groups))
w = 0.27
b1 = ax.bar(x - w, relevance_kappa, w, label='relevance', color=C['accent3'], edgecolor=C['accent3'])
b2 = ax.bar(x, stance_kappa, w, label='stance (전체)', color=C['accent2'], edgecolor=C['accent2'])
# stance rel>0만 있는 부분
xr = [1, 2]; yr = [0.920, 0.177]
b3 = ax.bar([1+w, 2+w], yr, w, label='stance (rel>0)', color=C['accent5'], edgecolor=C['accent5'])

# 값 표시
for bs in [b1, b2, b3]:
    for b in bs:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x()+b.get_width()/2, h+0.015, f'{h:.3f}',
                    ha='center', color=C['text'], fontsize=10, fontweight='bold')

ax.axhline(0.7, color=C['accent4'], linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2.5, 0.72, 'κ ≥ 0.7 기준', color=C['accent4'], fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(groups, color=C['text'], fontsize=11)
ax.set_ylabel("Cohen's κ", color=C['text'], fontsize=11)
ax.tick_params(axis='y', colors=C['subtext'])
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right', facecolor=C['panel'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=10)
ax.grid(axis='y', color=C['grid'], linestyle='--', alpha=0.3)

ax.text(1, -0.13, '사람 라벨러 일치도는 0.95+ │ Zero-shot 모델은 0.18~0.39 — fine-tune 필수성 데이터로 확인',
        ha='center', color=C['subtext'], fontsize=10, style='italic', transform=ax.transData)

plt.tight_layout()
plt.savefig('figures/F9_kappa_comparison.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F9_kappa_comparison.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F9")

# ========================================
# F10 — Dev 100 라벨 분포 (relevance × stance × 산업 stacked)
# ========================================
with open('splits/dev_labels_labelerA.csv', encoding='utf-8-sig') as f:
    labels = list(csv.DictReader(f))

# 산업 × stance 누적 막대 (relevance>0만)
ind_stance = defaultdict(lambda: defaultdict(int))
for row in labels:
    ind = row['industry']
    rel = row['relevance']
    st = row['stance_label']
    if rel != '0':
        ind_stance[ind][st] += 1

fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=C['bg'])

# 좌: relevance 분포
ax = axes[0]
setup_dark(ax, "Dev 100 · relevance 분포")
rel_count = Counter(r['relevance'] for r in labels)
ax.bar(['0 (무관)', '1 (배경)', '2 (직접)'],
       [rel_count['0'], rel_count['1'], rel_count['2']],
       color=[C['accent4'], C['accent2'], C['accent1']], edgecolor=C['grid'])
for i, lab in enumerate(['0', '1', '2']):
    ax.text(i, rel_count[lab]+1, rel_count[lab], ha='center', color=C['text'],
            fontsize=12, fontweight='bold')
ax.set_ylabel('건수', color=C['text'])
ax.tick_params(axis='both', colors=C['subtext'])

# 우: 산업 × stance stacked (rel>0만)
ax = axes[1]
setup_dark(ax, "Dev 100 · 산업 × stance (relevance>0)")
inds = ind_order
sup = [ind_stance[i]['support'] for i in inds]
neu = [ind_stance[i]['neutral'] for i in inds]
con = [ind_stance[i]['contradict'] for i in inds]

x = np.arange(len(inds))
ax.bar(x, sup, color=C['accent1'], label='support', edgecolor=C['grid'])
ax.bar(x, neu, bottom=sup, color=C['accent2'], label='neutral', edgecolor=C['grid'])
ax.bar(x, con, bottom=[s+n for s,n in zip(sup,neu)], color=C['accent4'], label='contradict', edgecolor=C['grid'])

# 값 표시
for i, (s, n, c) in enumerate(zip(sup, neu, con)):
    total = s + n + c
    if total > 0:
        ax.text(i, total+0.5, f'{total}', ha='center', color=C['text'], fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([i.split('/')[0] for i in inds], color=C['text'], rotation=20, ha='right', fontsize=10)
ax.set_ylabel('건수', color=C['text'])
ax.tick_params(axis='y', colors=C['subtext'])
ax.legend(facecolor=C['panel'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=10)

plt.tight_layout()
plt.savefig('figures/F10_dev100_label_dist.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F10_dev100_label_dist.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F10")

print("\nF6~F10 묶음 완료")
