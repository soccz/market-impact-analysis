"""
Figure F17~F24 생성 — 사고흐름 시각화 보강
F17: 의사결정 트리 (v0.1 → v2.11)
F18: 함정 7개 → 패치 매핑
F19: 라벨링 가이드 v1 → v1.1
F20: 정책 카드 v1.0 → v1.1 변화
F21: PTEI 식 5항목 다이어그램
F22: 정책별 PTEI/Contra 산점 (Plotly + PNG)
F23: 데이터 흐름 sankey (Plotly)
F24: Sanity 시리즈 timeline (Plotly + PNG)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

C = {
    'bg': '#fafaf9', 'panel': '#ffffff', 'text': '#1c1917', 'subtext': '#78716c',
    'accent1': '#2d9a6a', 'accent2': '#c66d3a', 'accent3': '#4f7ce6',
    'accent4': '#c44a64', 'accent5': '#8967b4', 'grid': '#e7e5e4',
    'yellow': '#c9961f',
}
PLOTLY_BG = '#fafaf9'

def setup_light(ax, title=None):
    ax.set_facecolor(C['bg'])
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title, color=C['text'], fontsize=14, fontweight='bold', pad=15)

# ========================================
# F17 — 의사결정 트리 (v0.1 → v2.11)
# ========================================
fig, ax = plt.subplots(figsize=(16, 10), facecolor=C['bg'])
setup_light(ax, "Figure 17 · 의사결정 트리 — v0.1에서 v2.11까지 14개 분기점")
ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis('off')

# 노드 정의: (x, y, label, type, color)
# type: 'start'·'pivot'·'patch'·'frozen'·'end'
nodes = [
    (1, 11, 'v0.1\n감성·부호정렬', 'start', C['subtext']),
    (3, 11, 'v1.0\nDirection Score', 'pivot', C['accent5']),
    (5, 11, 'v2.0\nPPT-NLP\n6채널 도입', 'pivot', C['accent5']),
    (7, 11, 'v2.1\n채널 NLI\n4유형 분류', 'frozen', C['accent1']),
    (9, 11, 'v2.2\n시간창\n2층 검정', 'frozen', C['accent1']),
    (11, 11, 'v2.3\n방향고정\n음수봉쇄', 'pivot', C['accent5']),
    (13, 11, 'v2.4\n19편 매핑\n고유변형 7', 'frozen', C['accent1']),
    (15, 11, 'v2.5\n결함 7 패치', 'patch', C['accent2']),

    (1, 7, 'v2.6\n과적합방지\nsanity/test', 'frozen', C['accent1']),
    (3, 7, 'v2.7\n정책카드 동결\nbuild_hypothesis', 'frozen', C['accent1']),
    (5, 7, 'v2.7.1\n책임분리\nNLI/gate', 'patch', C['accent2']),
    (7, 7, 'v2.7.2\nD-2 결과\n반영', 'patch', C['accent2']),
    (9, 7, 'v2.8\nv1.0 동결\nDev/Test', 'frozen', C['accent1']),
    (11, 7, 'v2.9\nD-4·D-5\nv1.1', 'patch', C['accent2']),
    (13, 7, 'v2.10\n가이드 v1.1', 'frozen', C['accent1']),
    (15, 7, 'v2.11\nDev 100 κ 0.95+\n1차 전수추론', 'end', C['accent4']),
]

# 주요 분기점 라벨 (아래에 의미)
branches = [
    (2, 9.8, '감성→방향'),
    (4, 9.8, '4축→채널'),
    (10, 9.8, '예측→설명'),
    (15, 9.8, '함정→패치'),
    (10, 5.8, '키워드→매핑'),
    (12, 5.8, 'gate 정제'),
]

for x, y, label, ntype, color in nodes:
    box = FancyBboxPatch((x-0.85, y-0.65), 1.7, 1.3, boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor=C['panel'], edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', color=color, fontsize=8.5, fontweight='bold')

# 화살표 (순차)
order = nodes
for i in range(len(order)-1):
    x1, y1 = order[i][0], order[i][1]
    x2, y2 = order[i+1][0], order[i+1][1]
    # 7→8 (1열 → 2열): 화살표 wrap
    if i == 7:  # 15→1 (행 바뀜)
        ax.annotate('', xy=(15.9, 10.2), xytext=(15.9, 11), arrowprops=dict(arrowstyle='-', color=C['subtext'], lw=1.2))
        ax.annotate('', xy=(1, 7.7), xytext=(1, 10.2), arrowprops=dict(arrowstyle='-', color=C['subtext'], lw=1.2))
        ax.plot([1, 15.9], [10.2, 10.2], color=C['subtext'], lw=1.2)
    else:
        ax.add_patch(FancyArrowPatch((x1+0.85, y1), (x2-0.85, y2),
                     arrowstyle='->,head_width=0.25', color=C['subtext'], linewidth=1.2))

for x, y, label in branches:
    ax.text(x, y, label, ha='center', color=C['accent2'], fontsize=8, fontweight='bold', style='italic')

# 범례
legend = [
    (C['subtext'], 'start'),
    (C['accent5'], 'pivot (방향 전환)'),
    (C['accent1'], 'frozen (동결)'),
    (C['accent2'], 'patch (함정 보정)'),
    (C['accent4'], 'end (현재 v2.11)'),
]
for i, (color, label) in enumerate(legend):
    cx = 1 + i*3
    ax.scatter(cx, 2.5, s=200, c=color, edgecolors=C['panel'], linewidth=2)
    ax.text(cx+0.4, 2.5, label, va='center', color=C['text'], fontsize=10)

ax.text(8, 0.8, '14개 버전 · 사용자/Claude 협업으로 진화한 사고의 흐름',
        ha='center', color=C['subtext'], fontsize=11, style='italic')

plt.tight_layout()
plt.savefig('figures/F17_decision_tree.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F17_decision_tree.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F17")

# ========================================
# F18 — 함정 7개 → 패치 매핑
# ========================================
fig, ax = plt.subplots(figsize=(16, 9), facecolor=C['bg'])
setup_light(ax, "Figure 18 · 함정 7개 → 패치 — 사용자 지적이 식으로 박힌 순간들")
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')

traps = [
    ('4.1', 'channel_match × NLI', '이중반영', 'multiplier → indicator gate', C['accent4']),
    ('4.2', 'novelty 재발', '이중반영 (또)', '기사 PTEI만, 일별 mean', C['accent4']),
    ('4.3', 'placebo window', '독립성 약함', '2-track (far + cross-policy)', C['accent2']),
    ('4.4', 'case별 가설', '과적합 위험', 'build_hypothesis() 자동화', C['accent5']),
    ('4.5', 'sanity 누적', '과적합 위험', 'sanity/dev/test 분리', C['accent5']),
    ('4.6', 'relevance gate', '규칙 누적', '4줄 동결', C['accent1']),
    ('4.7', 'subtype 5개 부족', 'B·D·E 혼재', '8개 type_subtype 동결', C['accent1']),
]

col_x = [0.5, 1.7, 4.5, 8, 12.5]
col_w = [1.0, 2.6, 3.2, 4.2, 3.2]
headers = ['§', '발견', '원인', '패치', '결과']
for h, x, w in zip(headers, col_x, col_w):
    box = FancyBboxPatch((x, 7.6), w, 0.6, boxstyle="round,pad=0.05,rounding_size=0.1",
                          facecolor=C['panel'], edgecolor=C['accent3'], linewidth=1.5)
    ax.add_patch(box)
    ax.text(x+w/2, 7.9, h, ha='center', va='center', color=C['accent3'], fontsize=10, fontweight='bold')

for ri, (sec, found, cause, patch, color) in enumerate(traps):
    y = 6.8 - ri*0.95
    cells = [sec, found, cause, patch, 'gate' if '4.1' in sec else 'PTEI' if '4.2' in sec else 'H3' if '4.3' in sec else '자동화' if '4.4' in sec else '원칙' if '4.5' in sec else '4줄' if '4.6' in sec else '8 subtype']
    cells = [sec, found, cause, patch, ['gate', 'PTEI 정합', 'H3 강건', '자동화 룰', '데이터 분리', 'gate 4줄', '8 subtype'][ri]]
    for ci, (val, x, w) in enumerate(zip(cells, col_x, col_w)):
        text_c = color if ci in (0, 4) else (C['accent4'] if ci == 1 else C['accent2'] if ci == 2 else C['text'])
        fw = 'bold' if ci in (0, 1, 4) else 'normal'
        box = FancyBboxPatch((x, y-0.35), w, 0.75, boxstyle="round,pad=0.03,rounding_size=0.08",
                              facecolor=C['panel'], edgecolor=C['grid'], linewidth=0.8)
        ax.add_patch(box)
        ax.text(x+w/2, y, val, ha='center', va='center', color=text_c, fontsize=9, fontweight=fw)

ax.text(8, 0.3, '7개 함정 모두 사용자가 검토 중에 짚어냈고, 각각 식·원칙으로 박혀 v2.11 완성',
        ha='center', color=C['subtext'], fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('figures/F18_traps_patches.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F18_traps_patches.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F18")

# ========================================
# F19 — 라벨링 가이드 v1 → v1.1 진화
# ========================================
fig, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
setup_light(ax, "Figure 19 · 라벨링 가이드 — v1 (5룰) → v1.1 (7룰 보강)")
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')

# v1 rules (좌)
ax.text(3.5, 7.3, 'v1 (초안, 5룰)', ha='center', color=C['accent3'], fontsize=12, fontweight='bold')
v1_rules = [
    '3.1 공매도 금지 그레이존',
    '3.2 의대증원 정원 발표',
    '3.3 외국 정책 산업별',
    '3.4 정보성 기사',
    '3.5 종목명 단독 등장',
]
for i, r in enumerate(v1_rules):
    y = 6.3 - i*0.7
    box = FancyBboxPatch((0.5, y-0.25), 6, 0.55, boxstyle="round,pad=0.03,rounding_size=0.08",
                          facecolor=C['panel'], edgecolor=C['accent3'], linewidth=1)
    ax.add_patch(box)
    ax.text(3.5, y, r, ha='center', va='center', color=C['text'], fontsize=10)

# 화살표
ax.annotate('', xy=(7.5, 4.5), xytext=(6.7, 4.5),
            arrowprops=dict(arrowstyle='->,head_width=0.3', color=C['accent2'], lw=2))
ax.text(7.1, 5, '파일럿 30\nκ 0.94+\n불일치 2건', ha='center', color=C['accent2'], fontsize=9, fontweight='bold')

# v1.1 rules (우, +2)
ax.text(10.5, 7.3, 'v1.1 (보강, 7룰)', ha='center', color=C['accent1'], fontsize=12, fontweight='bold')
v11_rules = [
    '3.1 공매도 금지 그레이존',
    '3.2 의대증원 정원 발표',
    '3.3 외국 정책 산업별',
    '3.4 정보성 기사',
    '3.5 종목명 단독 등장',
    '★ 3.6 동정·일정 기사 (신설)',
    '★ 3.7 배경 안정화 조치 (신설)',
]
for i, r in enumerate(v11_rules):
    y = 6.3 - i*0.55
    is_new = '★' in r
    box = FancyBboxPatch((7.5, y-0.2), 6, 0.45, boxstyle="round,pad=0.03,rounding_size=0.08",
                          facecolor=C['panel'], edgecolor=C['accent1'] if is_new else C['grid'],
                          linewidth=2 if is_new else 1)
    ax.add_patch(box)
    ax.text(10.5, y, r, ha='center', va='center',
            color=C['accent1'] if is_new else C['text'], fontsize=10,
            fontweight='bold' if is_new else 'normal')

ax.text(7, 0.7, 'pn=34 시밀러 가이드 (동정 기사) + pn=13 공매도 금지 (배경 안정) 두 불일치 사례를 일반 원칙으로 흡수',
        ha='center', color=C['subtext'], fontsize=10, style='italic')
ax.text(7, 0.2, 'Dev 100 사람-사람 κ relevance 0.968 · stance 0.946 (PASS)',
        ha='center', color=C['accent1'], fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/F19_guide_evolution.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F19_guide_evolution.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F19")

# ========================================
# F20 — 정책 카드 v1.0 → v1.1 변화 (D-4·D-5)
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C['bg'])

# D-4: C4 채널 키워드 정제
ax = axes[0]
setup_light(ax, "D-4 · C4 키워드 정제 효과")
policies = ['K칩스법\n1차', '일본\n수출규제', '공매도\n금지 3차']
before = [0.83, 1.00, 1.00]
after = [0.50, 0.83, 1.00]
x = np.arange(len(policies))
w = 0.35
ax.bar(x - w/2, before, w, label='v1.0 (정제 전)', color=C['accent4'], edgecolor=C['accent4'])
ax.bar(x + w/2, after, w, label='v1.1 (정제 후)', color=C['accent1'], edgecolor=C['accent1'])
for i in range(len(policies)):
    delta = after[i] - before[i]
    if delta < 0:
        ax.text(i, before[i]+0.04, f'Δ {delta*100:+.0f}%p', ha='center', color=C['accent2'], fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(policies, color=C['text'])
ax.set_ylabel('C4 활성률', color=C['text'])
ax.set_ylim(0, 1.2)
ax.tick_params(colors=C['subtext'])
ax.legend(facecolor=C['panel'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=10)
ax.grid(axis='y', color=C['grid'], alpha=0.3)

# D-5: subtype 활성 채널 재매핑
ax = axes[1]
setup_light(ax, "D-5 · subtype 활성 채널 변화")
subtypes = ['A_tax', 'B_reg-', 'B_stab+', 'C_indu', 'D_relief', 'D_shock', 'E_ben+', 'E_shock-']
v10 = [[1,0,0,0,1,0], [0,1,0,1,0,0], [0,0,0,0,1,1], [0,0,1,0,0,1],
       [0,0,0,1,1,1], [0,0,0,1,0,1], [0,1,1,0,0,0], [0,1,1,1,0,0]]
v11 = [[1,0,0,0,0,0], [0,1,0,1,0,0], [0,0,0,0,1,1], [0,0,1,0,0,1],
       [0,0,0,1,1,1], [0,0,0,1,0,1], [0,1,1,0,0,0], [0,0,1,1,0,0]]
# 변경된 subtype만 표시 (A_tax: C5 제거, E_foreign_shock: C2 제거)
changes = [
    ('A_tax_support', 'C5 제거', '비용 채널만 (간소화)'),
    ('E_foreign_shock', 'C2 제거', '공급망·규제 중심'),
]
ax.text(5, 6.5, '주요 매핑 변경 (사용자 지적 후)', ha='center', color=C['text'], fontsize=12, fontweight='bold')
for i, (sub, what, why) in enumerate(changes):
    y = 5.3 - i*1.8
    box = FancyBboxPatch((1, y-0.7), 8, 1.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                         facecolor=C['panel'], edgecolor=C['accent2'], linewidth=1.5)
    ax.add_patch(box)
    ax.text(1.5, y+0.3, sub, color=C['accent5'], fontsize=11, fontweight='bold')
    ax.text(1.5, y-0.1, what, color=C['accent4'], fontsize=10)
    ax.text(1.5, y-0.45, why, color=C['subtext'], fontsize=9.5, style='italic')

ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

plt.suptitle("Figure 20 · 정책 카드 v1.0 → v1.1 (D-4·D-5 적용)",
             color=C['text'], fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/F20_policy_card_evolution.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F20_policy_card_evolution.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F20")

# ========================================
# F21 — PTEI 식 5항목 의미 다이어그램
# ========================================
fig, ax = plt.subplots(figsize=(16, 8), facecolor=C['bg'])
setup_light(ax, "Figure 21 · PTEI 식 5항목 — 각 인자가 무엇을 가두는가")
ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis('off')

# 식 (상단)
ax.text(8, 7.2, r'$PTEI_{a,c} = relevance_a \times \mathbb{I}(channel\_match_{a,c} \geq \tau_c) \times p\_support_{a,c} \times specificity_a \times novelty_a$',
        ha='center', color=C['text'], fontsize=13)

items = [
    (1.5, 'relevance', 'L1·L2·L3 gate', '정책 직접/배경 4줄 원칙', '0~1.0', C['accent3']),
    (4.7, 'channel_match', '{0, 1} indicator', 'keyword OR semantic gate', 'binary', C['accent5']),
    (8, 'p_support', 'NLI stance', 'D_i 방향 지지 확률', '0~1.0', C['accent1']),
    (11.2, 'specificity', 'NER + 정규식', '종목·수치·기관·금액 매칭', '0.5~1.5', C['accent4']),
    (14.5, 'novelty', 'cos distance', '전일 기사 대비 새로움', '0~1.0', C['yellow']),
]

for x, name, method, desc, rng, color in items:
    # 인자 박스
    box = FancyBboxPatch((x-1.3, 3.0), 2.6, 3.0, boxstyle="round,pad=0.1,rounding_size=0.15",
                         facecolor=C['panel'], edgecolor=color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x, 5.6, name, ha='center', color=color, fontsize=12, fontweight='bold')
    ax.text(x, 5.0, method, ha='center', color=C['text'], fontsize=10, fontweight='bold')
    ax.text(x, 4.3, desc, ha='center', color=C['subtext'], fontsize=9, wrap=True)
    ax.text(x, 3.4, f'범위: {rng}', ha='center', color=color, fontsize=9, fontweight='bold')

# 화살표
for i in range(len(items)-1):
    x1 = items[i][0] + 1.3
    x2 = items[i+1][0] - 1.3
    ax.text((x1+x2)/2, 4.5, '×', ha='center', color=C['subtext'], fontsize=18, fontweight='bold')

# 곱셈 결과 (하단)
ax.text(8, 1.8, 'PTEI ≥ 0 (음수 봉쇄)',
        ha='center', color=C['accent1'], fontsize=14, fontweight='bold')
ax.text(8, 1.2, '5개 인자 모두 ≥ 0 → 곱이 0 이상 → 길 B 설명모델 보장',
        ha='center', color=C['subtext'], fontsize=10, style='italic')
ax.text(8, 0.4, '반박 기사는 별도 지표 Contradiction_{a,c} (구조 동일, p_contradict 사용)',
        ha='center', color=C['accent4'], fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('figures/F21_ptei_components.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F21_ptei_components.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F21")

# ========================================
# F22 — 정책별 PTEI vs Contradiction 산점도 (Plotly)
# ========================================
# Top 10 + 약간 더 (활성 100건 이상)
TOP_ALL = [
    ('일본 수출규제', 1884.755, 1812.970, 'E_foreign_shock', '-', 3618),
    ('공매도 금지 4차', 1324.621, 930.700, 'B_market_stabilizer', '+', 2340),
    ('8·2 대책', 1221.949, 842.131, 'B_restriction_negative', '-', 2100),
    ('100조 패키지', 829.040, 513.774, 'D_crisis_relief', '+', 1511),
    ('4·1 대책', 708.224, 396.285, 'A_tax_support', '+', 1557),
    ('의약분업', 619.723, 471.173, 'B_restriction_negative', '-', 1376),
    ('9·13 대책', 564.862, 540.303, 'B_restriction_negative', '-', 1443),
    ('8·31 대책', 541.264, 455.676, 'B_restriction_negative', '-', 1358),
    ('자본시장법', 405.385, 211.139, 'C_industrial_support', '+', 751),
    ('그린뉴딜', 360.928, 155.779, 'C_industrial_support', '+', 648),
]
subtype_colors = {
    'A_tax_support': '#c66d3a',
    'B_restriction_negative': '#c44a64',
    'B_market_stabilizer': '#2d9a6a',
    'C_industrial_support': '#4f7ce6',
    'D_crisis_relief': '#8967b4',
    'D_crisis_shock': '#1c1917',
    'E_foreign_benefit': '#c9961f',
    'E_foreign_shock': '#78716c',
}

fig = go.Figure()
for sub in subtype_colors:
    pts = [(n, p, c, s, d, ac) for n, p, c, s, d, ac in TOP_ALL if s == sub]
    if not pts: continue
    fig.add_trace(go.Scatter(
        x=[p[1] for p in pts], y=[p[2] for p in pts], mode='markers+text',
        text=[p[0] for p in pts], textposition='top center', textfont=dict(size=10, color=C['text']),
        marker=dict(size=[p[5]/50 for p in pts], color=subtype_colors[sub],
                    line=dict(width=2, color=C['panel']), sizemode='area', sizemin=8),
        name=sub,
        customdata=[[p[3], p[4], p[1]/p[2]] for p in pts],
        hovertemplate='%{text}<br>PTEI %{x:.0f} · Contra %{y:.0f}<br>D_i %{customdata[1]} · 비율 %{customdata[2]:.2f}<extra></extra>',
    ))

# y=x 기준선
mx = max(p[1] for p in TOP_ALL) * 1.05
fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode='lines',
                          line=dict(dash='dash', color=C['accent4'], width=1.5),
                          name='y=x (균등)', hoverinfo='skip'))

fig.update_layout(
    title=dict(text="Figure 22 · 정책별 PTEI vs Contradiction — 거품 크기 = active 기사 수",
               font=dict(color=C['text'], size=16)),
    xaxis_title='PTEI sum', yaxis_title='Contradiction sum',
    paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
    font=dict(color=C['text']),
    legend=dict(bgcolor=C['panel'], bordercolor=C['grid'], font=dict(color=C['text'], size=9)),
    height=650, margin=dict(l=80, r=40, t=80, b=80),
)
fig.update_xaxes(gridcolor=C['grid'], color=C['subtext'])
fig.update_yaxes(gridcolor=C['grid'], color=C['subtext'])
fig.write_html('figures/F22_ptei_contra_scatter.html', include_plotlyjs='cdn')

# PNG 백업
fig_mpl, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
setup_light(ax, "Figure 22 · 정책별 PTEI vs Contradiction 산점")
for sub, color in subtype_colors.items():
    pts = [(n, p, c, s, d, ac) for n, p, c, s, d, ac in TOP_ALL if s == sub]
    if not pts: continue
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]; sizes = [p[5]/3 for p in pts]
    ax.scatter(xs, ys, s=sizes, c=color, alpha=0.75, edgecolors=C['panel'], linewidth=2, label=sub)
    for p in pts:
        ax.annotate(p[0], (p[1], p[2]), xytext=(5,5), textcoords='offset points',
                    color=C['text'], fontsize=9)
xs_line = [0, mx]
ax.plot(xs_line, xs_line, '--', color=C['accent4'], lw=1.5, alpha=0.7, label='y=x')
ax.set_xlabel('PTEI sum', color=C['text'])
ax.set_ylabel('Contradiction sum', color=C['text'])
ax.tick_params(colors=C['subtext'])
ax.legend(facecolor=C['panel'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=8, loc='upper left')
ax.grid(color=C['grid'], alpha=0.3)
plt.tight_layout()
plt.savefig('figures/F22_ptei_contra_scatter.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F22_ptei_contra_scatter.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F22 (Plotly + PNG)")

# ========================================
# F23 — 데이터 흐름 sankey (Plotly)
# ========================================
labels = [
    '99,539 기사',           # 0
    'relevance ≥ 0.4 (53,568)',  # 1
    'NO_MATCH (45,971)',     # 2
    'channel gate 통과 (28,603)', # 3
    'gate 탈락 (24,965)',     # 4
    'PTEI > 0 · D_i 지지 (PTEI 13,761)',  # 5
    'Contradiction > 0 (9,925)',  # 6
    'NLI neutral (76,342)',       # 7
]
source = [0, 0, 1, 1, 3, 3]
target = [1, 2, 3, 4, 5, 6]
value = [53568, 45971, 28603, 24965, 16445, 6752]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=18, thickness=20, line=dict(color=C['grid'], width=0.5),
        label=labels,
        color=[C['accent3'], C['accent1'], C['accent4'],
               C['accent2'], C['subtext'], C['accent1'], C['accent4'], C['subtext']],
    ),
    link=dict(
        source=source, target=target, value=value,
        color=['rgba(45,154,106,0.35)', 'rgba(196,74,100,0.25)',
               'rgba(198,109,58,0.35)', 'rgba(120,113,108,0.25)',
               'rgba(45,154,106,0.35)', 'rgba(196,74,100,0.35)'],
    )
)])
fig.update_layout(
    title=dict(text="Figure 23 · 데이터 흐름 — 99,539건 기사가 PTEI로 가는 과정",
               font=dict(color=C['text'], size=16)),
    paper_bgcolor=PLOTLY_BG, font=dict(color=C['text'], size=11),
    height=600, margin=dict(l=20, r=20, t=80, b=60),
)
fig.write_html('figures/F23_data_flow_sankey.html', include_plotlyjs='cdn')
print("✓ F23 (Plotly Sankey)")

# F23 PNG 백업 (단순 표 형태)
fig_mpl, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
setup_light(ax, "Figure 23 · 데이터 흐름 (99,539 → PTEI)")
stages = [
    (99539, '입력 기사', C['accent3']),
    (53568, 'relevance ≥ 0.4', C['accent1']),
    (28603, 'channel gate 활성', C['accent2']),
    (16445, 'PTEI support', C['accent1']),
]
xs = [0, 1, 2, 3]
ys = [s[0] for s in stages]
ax.bar(xs, ys, color=[s[2] for s in stages], width=0.65, edgecolor=C['panel'])
for x, (n, name, color) in zip(xs, stages):
    ax.text(x, n+1500, f'{n:,}', ha='center', color=color, fontsize=12, fontweight='bold')
    ax.text(x, -3500, name, ha='center', color=C['text'], fontsize=11)
ax.set_xticks([]); ax.tick_params(axis='y', colors=C['subtext'])
ax.set_ylim(-6000, 105000)
plt.tight_layout()
plt.savefig('figures/F23_data_flow_sankey.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F23_data_flow_sankey.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()

# ========================================
# F24 — Sanity 시리즈 timeline (Plotly)
# ========================================
events = [
    ('v3.1 relevance gate', 1, 'PASS', '10건 noise 4/4 정확 IRR', C['accent1']),
    ('C-1 채널 NLI', 2, 'FAIL 4/4', 'NLI 단독 채널 분리 실패\n→ 책임 분리 결정', C['accent4']),
    ('D-2 hybrid gate', 3, 'PASS 3/4', '공매도 금지 C5·C6 활성\n→ 표면감성 우회 증명', C['accent1']),
    ('v1.1 D-4·D-5', 4, '개선', 'K칩스법 C4 33%p ↓\n→ 키워드 정제 효과', C['accent2']),
    ('파일럿 30 사람-AI κ', 5, '0.94+', '가이드 v1 작동성 확인\n불일치 2건 → v1.1 보강', C['accent1']),
    ('Dev 100 사람-사람 κ', 6, '0.95+', '정식 라벨 채택 가능', C['accent1']),
    ('Zero-shot baseline', 7, 'κ 0.18~0.39', 'fine-tune 필요성 데이터로 확인', C['accent4']),
    ('1차 전수 추론', 8, 'PTEI/Contra 1.39', '99K 추론 완료 (탐색용)', C['accent2']),
]
fig = go.Figure()
for name, x, result, desc, color in events:
    fig.add_trace(go.Scatter(
        x=[x], y=[1], mode='markers+text',
        marker=dict(size=24, color=color, line=dict(color=C['panel'], width=2)),
        text=[result], textposition='top center', textfont=dict(color=color, size=11, family='Noto Serif KR'),
        name=name, hovertext=f'{name}<br>{desc}', hoverinfo='text',
        showlegend=False,
    ))
    fig.add_annotation(text=name, x=x, y=0.65, showarrow=False,
                       font=dict(color=C['text'], size=10))
    fig.add_annotation(text=desc, x=x, y=0.3, showarrow=False,
                       font=dict(color=C['subtext'], size=9))

# 연결선
fig.add_trace(go.Scatter(x=[1,8], y=[1,1], mode='lines',
                          line=dict(color=C['grid'], width=2), showlegend=False, hoverinfo='skip'))

fig.update_layout(
    title=dict(text="Figure 24 · Sanity 시리즈 timeline — 실패와 결정의 사슬",
               font=dict(color=C['text'], size=16)),
    paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
    height=420, margin=dict(l=40, r=40, t=80, b=80),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0.3, 8.7]),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1.5]),
    showlegend=False,
)
fig.write_html('figures/F24_sanity_timeline.html', include_plotlyjs='cdn')
print("✓ F24 (Plotly)")

# F24 PNG 백업
fig_mpl, ax = plt.subplots(figsize=(16, 6), facecolor=C['bg'])
setup_light(ax, "Figure 24 · Sanity 시리즈 timeline — 실패와 결정의 사슬")
ax.set_xlim(0, 9); ax.set_ylim(0, 4); ax.axis('off')
ax.plot([0.5, 8.5], [2, 2], color=C['grid'], lw=2.5)
for name, x, result, desc, color in events:
    ax.scatter(x, 2, s=600, c=color, edgecolors=C['panel'], linewidth=2.5, zorder=3)
    ax.text(x, 2.55, name, ha='center', color=C['text'], fontsize=9.5, fontweight='bold')
    ax.text(x, 1.5, result, ha='center', color=color, fontsize=9, fontweight='bold')
    ax.text(x, 1.0, desc, ha='center', color=C['subtext'], fontsize=8)

plt.tight_layout()
plt.savefig('figures/F24_sanity_timeline.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F24_sanity_timeline.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()

print("\nF17~F24 완료 (총 8개, PNG 8 + WebP 8 + Plotly HTML 3)")
