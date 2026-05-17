"""
Figure F1~F5 생성 — Part 1~4 개념도·매트릭스·식 변천
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 컬러 팔레트 (higan 스타일 영감 — 다크 + 액센트)
C = {
    'bg': '#1a1d24',
    'panel': '#252932',
    'text': '#e6e9ef',
    'subtext': '#8a93a6',
    'accent1': '#5fbf8f',  # green (성공)
    'accent2': '#e88c5a',  # orange (전환)
    'accent3': '#7a9cf2',  # blue (정보)
    'accent4': '#d96e85',  # red (실패·경고)
    'accent5': '#b89adb',  # purple (pivot)
    'grid': '#363b46',
}

def setup_dark(ax, title=None):
    ax.set_facecolor(C['bg'])
    for spine in ax.spines.values(): spine.set_visible(False)
    if title:
        ax.set_title(title, color=C['text'], fontsize=14, fontweight='bold', pad=15)

# ========================================
# F1 — 프로젝트 3트랙 다이어그램
# ========================================
fig, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
setup_dark(ax, "Figure 1 · 프로젝트 3트랙 — CAR · NLP · 통합 보도")
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')

# 3개 트랙 박스
tracks = [
    (1, 4, '1차 트랙 — CAR 검증', '50건 정책 발표일 전후\n누적초과수익률 산출',
     'FDR 33/33 PASS\nNaver 17/17 PASS\n방향 50/50 PASS',
     'soccz/market-impact-analysis', C['accent3']),
    (5.5, 4, '2차 트랙 — NLP (PTEI)', '99,539건 기사에서\n정책 전달경로 증거 측정',
     'Dev 100 κ 0.95+\n1차 전수 추론 완료\nfine-tune 대기',
     '본 보고서가 다루는 트랙', C['accent2']),
    (10, 4, '3차 트랙 — 통합 보도', 'CAR × PTEI 결합 검정\n+ 5부작 기사 + 시각화',
     '대기',
     '한국언론진흥재단\n2026 기획취재 사업',
     C['accent5']),
]

for x, y, title, what, status, who, color in tracks:
    # 본문 박스
    box = FancyBboxPatch((x, y-1.8), 3, 3.6, boxstyle="round,pad=0.1,rounding_size=0.2",
                          facecolor=C['panel'], edgecolor=color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x+1.5, y+1.4, title, ha='center', va='top', color=color,
            fontsize=12, fontweight='bold')
    ax.text(x+1.5, y+0.6, what, ha='center', va='top', color=C['text'], fontsize=9.5)
    ax.text(x+1.5, y-0.4, status, ha='center', va='top', color=C['accent1'], fontsize=9)
    ax.text(x+1.5, y-1.5, who, ha='center', va='top', color=C['subtext'], fontsize=8, style='italic')

# 화살표 (1차 → 2차 → 3차)
for x in [4, 8.5]:
    arrow = FancyArrowPatch((x, 4), (x+1.5, 4), arrowstyle='->,head_width=0.4,head_length=0.5',
                             color=C['text'], linewidth=2)
    ax.add_patch(arrow)

# 핵심 질문 (상단)
ax.text(7, 7.3, '"정책은 자본시장을 얼마나 흔들었나, 언론은 그 사이 어떤 역할이었나"',
        ha='center', color=C['accent2'], fontsize=12, fontweight='bold', style='italic')

# 데이터 자산 (하단)
ax.text(7, 0.5, '데이터 자산  ·  50건 정책 (1998~2025)  ·  99,539건 기사  ·  102 언론사  ·  본문 200자cap',
        ha='center', color=C['subtext'], fontsize=10)

plt.tight_layout()
plt.savefig('figures/F1_three_tracks.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F1_three_tracks.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F1")

# ========================================
# F2 — 200자 한계 + 표면 어휘 함정 매트릭스
# ========================================
fig, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
setup_dark(ax, "Figure 2 · 200자 본문 한계 + 표면 어휘 함정")
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

# 3개 사례 매트릭스
cases = [
    ('공매도 금지 (2008·2020)', '"공매도 전면 금지"\n"규제 강화"',
     '단어 표면 → 부정',
     '리먼·코로나 직후\n시장 안정 호재',
     '실제 D_i = +1 (호재)',
     '단순 NLI: contradict\n(잘못)'),
    ('일본 수출규제 (2019)', '"수출규제 발동"\n"화이트리스트 제외"',
     '단어 표면 → 부정',
     '삼성·SK 직격\n소부장 호재',
     '대상 종목별 정반대',
     '단순 NLI: 한쪽만 잡음'),
    ('K칩스법 (2023)', '"세액공제 8→15%"\n"법안 발의·심사"',
     '본문은 행정 절차',
     '비용 절감·투자 확대',
     '실제 D_i = +1 (호재)',
     '단순 NLI: 중립\n(약함)'),
]

col_titles = ['정책', '본문 (200자)', '단어 표면 판단', '실제 시장 효과', 'D_i 정답', 'NLI 한계']
col_x = [0.5, 2.7, 5.4, 7.8, 10.3, 12.2]
col_w = [2.2, 2.7, 2.4, 2.5, 1.9, 1.8]

# 헤더
for i, (title, x, w) in enumerate(zip(col_titles, col_x, col_w)):
    box = FancyBboxPatch((x, 5.6), w, 0.6, boxstyle="round,pad=0.05,rounding_size=0.1",
                          facecolor=C['panel'], edgecolor=C['accent3'], linewidth=1.5)
    ax.add_patch(box)
    ax.text(x+w/2, 5.9, title, ha='center', va='center', color=C['accent3'],
            fontsize=10, fontweight='bold')

# 행
row_colors = [C['accent4'], C['accent2'], C['accent4']]
for ri, (policy, body, surface, real, d_i, limit) in enumerate(cases):
    y = 4.6 - ri*1.6
    cells = [policy, body, surface, real, d_i, limit]
    for ci, (val, x, w) in enumerate(zip(cells, col_x, col_w)):
        # 컬럼별 색상 차별화
        if ci == 0: text_c = C['accent5']; fw = 'bold'
        elif ci == 2: text_c = C['accent4']; fw = 'normal'  # 표면 → 빨강
        elif ci == 3: text_c = C['accent1']; fw = 'normal'  # 실제 효과 → 초록
        elif ci == 4: text_c = row_colors[ri]; fw = 'bold'
        elif ci == 5: text_c = C['accent4']; fw = 'normal'
        else: text_c = C['text']; fw = 'normal'
        ax.text(x+w/2, y, val, ha='center', va='center', color=text_c,
                fontsize=8.5, fontweight=fw)

ax.text(7, 0.3, '함의: 단순 감성분석/NLI는 정책 도메인의 그레이존을 못 잡는다 → 정책 전달경로 측정으로 전환',
        ha='center', color=C['accent2'], fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('figures/F2_surface_trap.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F2_surface_trap.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F2")

# ========================================
# F3 — 길 A vs 길 B 개념도
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=C['bg'])
for ax in axes: setup_dark(ax); ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')

# 길 A
ax = axes[0]
ax.set_title("길 A — 독립 예측형 (탈락)", color=C['accent4'], fontsize=13, fontweight='bold', pad=10)
# 박스들: 뉴스 → NLP → 주가
boxes_a = [
    (1, 5.5, 2.5, 1.5, '뉴스 기사', C['accent3']),
    (5, 5.5, 2.5, 1.5, 'NLP (감성)', C['accent2']),
    (1, 2, 6.5, 1.5, '주가 방향?', C['accent5']),
]
for x, y, w, h, text, color in boxes_a:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                          facecolor=C['panel'], edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=color, fontsize=11, fontweight='bold')
# 화살표
ax.add_patch(FancyArrowPatch((3.5, 6.25), (5, 6.25), arrowstyle='->,head_width=0.3',
             color=C['text'], linewidth=2))
ax.add_patch(FancyArrowPatch((6.25, 5.5), (4.25, 3.5), arrowstyle='->,head_width=0.3',
             color=C['accent4'], linewidth=2.5))
ax.text(6.5, 4.5, '예측', color=C['accent4'], fontsize=10, fontweight='bold')
ax.text(5, 0.8, '50건 방향 일치 보장 X\n(자연 발생에 의존)',
        ha='center', color=C['accent4'], fontsize=10, style='italic')

# 길 B
ax = axes[1]
ax.set_title("길 B — 방향 고정 설명형 (채택)", color=C['accent1'], fontsize=13, fontweight='bold', pad=10)
boxes_b = [
    (1, 5.5, 6.5, 1.5, '검증된 주가 방향 D_i (앵커)', C['accent1']),
    (1, 2, 2.5, 1.5, '뉴스 기사', C['accent3']),
    (5, 2, 2.5, 1.5, 'NLP가 근거 채굴', C['accent2']),
]
for x, y, w, h, text, color in boxes_b:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                          facecolor=C['panel'], edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=color, fontsize=11, fontweight='bold')
ax.add_patch(FancyArrowPatch((4.25, 5.5), (6.25, 3.5), arrowstyle='->,head_width=0.3',
             color=C['accent1'], linewidth=2.5))
ax.add_patch(FancyArrowPatch((3.5, 2.75), (5, 2.75), arrowstyle='->,head_width=0.3',
             color=C['text'], linewidth=2))
ax.text(7, 4.5, '설명', color=C['accent1'], fontsize=10, fontweight='bold')
ax.text(5, 0.8, '50건 방향 일치 설계 보장\n검정은 강도·타이밍·위약',
        ha='center', color=C['accent1'], fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('figures/F3_path_a_vs_b.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F3_path_a_vs_b.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F3")

# ========================================
# F4 — 방향 고정 설명모델 다이어그램 + PTEI 식
# ========================================
fig, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
setup_dark(ax, "Figure 4 · 방향 고정 설명모델 + PTEI 식 구조")
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')

# D_i 앵커 (상단)
box = FancyBboxPatch((4, 6.3), 6, 1.2, boxstyle="round,pad=0.1,rounding_size=0.2",
                     facecolor=C['panel'], edgecolor=C['accent1'], linewidth=2.5)
ax.add_patch(box)
ax.text(7, 6.9, 'D_i = sign(CAR_{i, [0, k_i]})  (서강대 검증 산출물, NLP 추정 X)',
        ha='center', va='center', color=C['accent1'], fontsize=12, fontweight='bold')

# 화살표 (앵커 → PTEI 식)
ax.add_patch(FancyArrowPatch((7, 6.2), (7, 5.5), arrowstyle='->,head_width=0.4',
             color=C['text'], linewidth=2))

# PTEI 식 (중앙)
box = FancyBboxPatch((1, 3.5), 12, 1.8, boxstyle="round,pad=0.1,rounding_size=0.2",
                     facecolor=C['panel'], edgecolor=C['accent2'], linewidth=2.5)
ax.add_patch(box)
ax.text(7, 4.7, 'PTEI 식 (v2.7.1 동결)', ha='center', color=C['accent2'], fontsize=11, fontweight='bold')
ax.text(7, 4.0, r'$PTEI_{a,c} = relevance_a \times \mathbb{I}(channel\_match_{a,c} \geq \tau_c) \times p\_support_{a,c} \times specificity_a \times novelty_a$',
        ha='center', color=C['text'], fontsize=13)

# 5개 항목 설명 (하단)
items = [
    (1.2, 'relevance', 'L1·L2·L3 gate\n4줄 원칙', C['accent3']),
    (3.7, 'channel_match', 'keyword + 임베딩\nindicator gate', C['accent5']),
    (6.4, 'p_support', 'NLI stance\n(KLUE-NLI)', C['accent1']),
    (9.0, 'specificity', 'NER + 정규식\n(종목·수치·기관)', C['accent4']),
    (11.6, 'novelty', '1 - cos(D-1 기사)\n(중복 보도 차단)', C['accent2']),
]
for x, name, desc, color in items:
    box = FancyBboxPatch((x-0.9, 1.2), 1.8, 1.6, boxstyle="round,pad=0.05,rounding_size=0.1",
                          facecolor=C['panel'], edgecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, 2.45, name, ha='center', color=color, fontsize=10, fontweight='bold')
    ax.text(x, 1.7, desc, ha='center', color=C['text'], fontsize=8.5)

ax.text(7, 0.4, '음수 봉쇄 (≥ 0) · 반박은 Contradiction 별도 지표 · 방향 일치는 설계 보장 · 검정은 강도·타이밍·위약',
        ha='center', color=C['subtext'], fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('figures/F4_explanatory_model.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F4_explanatory_model.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F4")

# ========================================
# F5 — PTEI 식 변천 3단
# ========================================
fig, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
setup_dark(ax, "Figure 5 · PTEI 식 변천 — 함정 발견과 패치")
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')

stages = [
    (1, 5.5, 'v2.0 — 첫 식 (4 factor multiplier)',
     r'$stance_{a,c} \times channel\_match_{a,c} \times relevance_a \times specificity_a$',
     'channel_match가 NLI와 이중반영 위험\nstance ± 값으로 길 A 흔적',
     C['accent4']),
    (1, 3.3, 'v2.5 — channel_match를 gate로 강등',
     r'$relevance_a \times gate(a,c) \times stance_{a,c} \times specificity_a \times novelty_a$',
     '책임 분리 시작\n단 gate가 multiplier로 표현되어 모호',
     C['accent2']),
    (1, 1.1, 'v2.7.1 — indicator gate + 음수 봉쇄 (최종)',
     r'$relevance_a \times \mathbb{I}(channel\_match_{a,c} \geq \tau_c) \times p\_support_{a,c} \times specificity_a \times novelty_a$',
     'gate가 indicator {0,1}로 명문화\nstance는 p_support ≥ 0 (음수 봉쇄, Contradiction 분리)',
     C['accent1']),
]

for x, y, title, eq, note, color in stages:
    box = FancyBboxPatch((x, y-0.5), 12, 1.8, boxstyle="round,pad=0.1,rounding_size=0.2",
                         facecolor=C['panel'], edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+0.3, y+1.0, title, color=color, fontsize=11, fontweight='bold')
    ax.text(x+6, y+0.5, eq, ha='center', color=C['text'], fontsize=11)
    ax.text(x+0.3, y-0.2, note, color=C['subtext'], fontsize=9, va='top')

# 화살표
ax.add_patch(FancyArrowPatch((7, 5.0), (7, 4.4), arrowstyle='->,head_width=0.4',
             color=C['text'], linewidth=2))
ax.add_patch(FancyArrowPatch((7, 2.8), (7, 2.2), arrowstyle='->,head_width=0.4',
             color=C['text'], linewidth=2))

ax.text(7, 0.3, '함정 발견 → 패치 → 동결의 흐름. 식 자체가 결정의 기록이다.',
        ha='center', color=C['subtext'], fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('figures/F5_ptei_evolution.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F5_ptei_evolution.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F5")

print("\nF1~F5 묶음 완료 (각 PNG + WebP)")
