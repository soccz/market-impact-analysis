"""
Figure F11~F16 생성 — 전수 추론 결과·진행 상태·자산 매트릭스
F11·F13·F15 = Plotly 인터랙티브 HTML
F12·F14·F16 = matplotlib PNG/WebP
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import csv
import json
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

C = {
    'bg': '#1a1d24', 'panel': '#252932', 'text': '#e6e9ef', 'subtext': '#8a93a6',
    'accent1': '#5fbf8f', 'accent2': '#e88c5a', 'accent3': '#7a9cf2',
    'accent4': '#d96e85', 'accent5': '#b89adb', 'grid': '#363b46',
}
PLOTLY_BG = '#1a1d24'
PLOTLY_PANEL = '#252932'

def setup_dark(ax, title=None):
    ax.set_facecolor(C['bg'])
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title, color=C['text'], fontsize=14, fontweight='bold', pad=15)

# ========================================
# 데이터 로드 — article_scores 통계 (summary md 사용)
# ========================================
TOP_POLICIES = [
    ('일본 수출규제', 5000, 3618, 1884.755, 1812.970, 'E_foreign_shock', '-'),
    ('공매도 금지 4차', 3231, 2340, 1324.621, 930.700, 'B_market_stabilizer', '+'),
    ('8·2 대책', 5000, 2100, 1221.949, 842.131, 'B_restriction_negative', '-'),
    ('100조 패키지', 5000, 1511, 829.040, 513.774, 'D_crisis_relief', '+'),
    ('4·1 대책', 4776, 1557, 708.224, 396.285, 'A_tax_support', '+'),
    ('의약분업', 4804, 1376, 619.723, 471.173, 'B_restriction_negative', '-'),
    ('9·13 대책', 3443, 1443, 564.862, 540.303, 'B_restriction_negative', '-'),
    ('8·31 대책', 3952, 1358, 541.264, 455.676, 'B_restriction_negative', '-'),
    ('자본시장법', 2229, 751, 405.385, 211.139, 'C_industrial_support', '+'),
    ('그린뉴딜', 5000, 648, 360.928, 155.779, 'C_industrial_support', '+'),
]
CHANNEL_DATA = [
    ('C1 비용/세부담', 3647, 1422.730, 1011.747),
    ('C2 수요/매출', 4744, 1678.420, 1332.175),
    ('C3 공급/생산능력', 7090, 2585.142, 1761.073),
    ('C4 규제/불확실성', 12793, 4152.368, 3369.036),
    ('C5 자금조달/유동성', 4900, 1760.089, 1227.142),
    ('C6 시장신뢰/투자심리', 5208, 2162.797, 1224.721),
]
GATE_LEVELS = [
    ('NO_MATCH', 45971),
    ('L1_primary_exact', 32975),
    ('L2_primary_token_AND', 15974),
    ('L1.5_secondary_exact', 2134),
    ('L3_alias', 1456),
    ('L2.5_secondary_token_AND', 1029),
]

# ========================================
# F11 — Plotly 인터랙티브: 정책별 PTEI 상위 막대
# ========================================
names = [p[0] for p in TOP_POLICIES]
ptei_sum = [p[3] for p in TOP_POLICIES]
contra_sum = [p[4] for p in TOP_POLICIES]
ratio = [p/c if c > 0 else 0 for p, c in zip(ptei_sum, contra_sum)]
subtypes = [p[5] for p in TOP_POLICIES]
d_signs = [p[6] for p in TOP_POLICIES]

# 비율에 따른 색상
def ratio_color(r):
    if r >= 1.8: return C['accent1']
    if r >= 1.3: return C['accent2']
    if r >= 1.1: return C['accent5']
    return C['accent4']

colors = [ratio_color(r) for r in ratio]

fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                    subplot_titles=("PTEI sum (정책별 상위 10)", "PTEI / Contradiction 비율"),
                    horizontal_spacing=0.12)

fig.add_trace(go.Bar(
    y=names[::-1], x=ptei_sum[::-1], orientation='h',
    marker=dict(color=[C['accent3']]*10),
    name='PTEI sum',
    text=[f'{p:.0f}' for p in ptei_sum[::-1]],
    textposition='outside',
    hovertemplate='%{y}<br>PTEI: %{x:.1f}<br>D_i: %{customdata[0]} · %{customdata[1]}<extra></extra>',
    customdata=list(zip(d_signs[::-1], subtypes[::-1])),
), row=1, col=1)

fig.add_trace(go.Bar(
    y=names[::-1], x=ratio[::-1], orientation='h',
    marker=dict(color=colors[::-1]),
    name='PTEI/Contra',
    text=[f'{r:.2f}' for r in ratio[::-1]],
    textposition='outside',
    hovertemplate='%{y}<br>비율: %{x:.2f}<extra></extra>',
), row=1, col=2)

fig.add_vline(x=1.0, line_dash="dash", line_color=C['accent4'], row=1, col=2)
fig.add_annotation(text="ratio = 1.0 (균등)", x=1.0, y=-0.5,
                   showarrow=False, font=dict(color=C['accent4'], size=10), row=1, col=2)

fig.update_layout(
    title=dict(text="Figure 11 · 정책별 PTEI 상위 10 + Contradiction 비율 (1차 전수 추론)",
               font=dict(color=C['text'], size=16)),
    paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
    font=dict(color=C['text']), showlegend=False,
    height=600, margin=dict(l=140, r=80, t=80, b=60),
)
fig.update_xaxes(gridcolor=C['grid'], showline=False, zeroline=False, color=C['subtext'])
fig.update_yaxes(gridcolor=C['grid'], showline=False, color=C['text'])

fig.write_html('figures/F11_policy_ptei_top10.html', include_plotlyjs='cdn')
print("✓ F11 (Plotly HTML)")

# PNG 백업도 생성
fig_mpl, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
setup_dark(ax, "Figure 11 · 정책별 PTEI 상위 10 + Contradiction 비율")
y_pos = np.arange(len(names))[::-1]
ax.barh(y_pos, ptei_sum, color=C['accent3'], height=0.5, label='PTEI sum')
ax.barh(y_pos - 0.3, [r*200 for r in ratio], color=[ratio_color(r) for r in ratio],
        height=0.25, label='PTEI/Contra × 200 (참고)')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, color=C['text'])
for i, (p, r) in enumerate(zip(ptei_sum, ratio)):
    ax.text(p+30, y_pos[i], f'PTEI {p:.0f} · 비율 {r:.2f}',
            va='center', color=C['text'], fontsize=9)
ax.tick_params(axis='x', colors=C['subtext'])
ax.legend(facecolor=C['panel'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=10)
ax.grid(axis='x', color=C['grid'], linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/F11_policy_ptei_top10.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F11_policy_ptei_top10.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()

# ========================================
# F12 — 6채널 PTEI sum 도넛 + gate count
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=C['bg'])

ch_names = [c[0] for c in CHANNEL_DATA]
gate_counts = [c[1] for c in CHANNEL_DATA]
ch_ptei = [c[2] for c in CHANNEL_DATA]
ch_colors = [C['accent3'], C['accent2'], C['accent1'], C['accent4'], C['accent5'], '#f0bb50']

# 좌: gate count
ax = axes[0]
setup_dark(ax, "6채널 · gate 통과 기사 수")
ax.pie(gate_counts, labels=ch_names, colors=ch_colors,
       autopct=lambda p: f'{p:.0f}%\n({int(p*sum(gate_counts)/100):,})',
       startangle=90, wedgeprops=dict(width=0.4, edgecolor=C['bg'], linewidth=2),
       textprops=dict(color=C['text'], fontsize=9))

# 우: PTEI sum
ax = axes[1]
setup_dark(ax, "6채널 · PTEI sum")
ax.pie(ch_ptei, labels=ch_names, colors=ch_colors,
       autopct=lambda p: f'{p:.0f}%\n({p*sum(ch_ptei)/100:,.0f})',
       startangle=90, wedgeprops=dict(width=0.4, edgecolor=C['bg'], linewidth=2),
       textprops=dict(color=C['text'], fontsize=9))

fig.suptitle("Figure 12 · 6채널 분포 — C4 규제/불확실성 dominance 잔존",
             color=C['text'], fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/F12_channel_distribution.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F12_channel_distribution.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F12")

# ========================================
# F13 — Plotly 인터랙티브: 일별 PTEI 시계열 샘플 (4정책)
# ========================================
with open('data/processed/daily_ptei_panel.csv', encoding='utf-8-sig') as f:
    panel = list(csv.DictReader(f))

print(f"  panel cols: {list(panel[0].keys())[:10]}")

# 4개 대표 정책 pn 매핑
target = {8: 'K칩스법 1차', 15: '공매도 금지 3차', 3: '일본 수출규제', 17: '100조 패키지'}

fig = go.Figure()

# panel 컬럼 — daily_ptei_panel.csv 실제 구조
ptei_col = 'PTEI_sum'
contra_col = 'Contradiction_sum'
pn_col = 'policy_n'
rd_col = 'rel_day'

if pn_col and rd_col and ptei_col:
    series_colors = [C['accent1'], C['accent2'], C['accent4'], C['accent3']]
    for i, (pn, name) in enumerate(target.items()):
        sub = [r for r in panel if str(r[pn_col]) == str(pn)]
        sub.sort(key=lambda r: int(r[rd_col]))
        xs = [int(r[rd_col]) for r in sub]
        ys_p = [float(r[ptei_col] or 0) for r in sub]

        fig.add_trace(go.Scatter(
            x=xs, y=ys_p, mode='lines+markers',
            name=f'pn={pn} {name}', line=dict(color=series_colors[i], width=2.5),
            marker=dict(size=5),
            hovertemplate=f'{name}<br>rel_day=%{{x}}<br>PTEI=%{{y:.2f}}<extra></extra>',
        ))

    fig.add_vline(x=0, line_dash="dash", line_color=C['accent5'], opacity=0.6)
    fig.add_annotation(text="D=0 (정책 발표일)", x=0, y=1.02, yref='paper',
                       showarrow=False, font=dict(color=C['accent5'], size=11))

fig.update_layout(
    title=dict(text="Figure 13 · 일별 PTEI 시계열 — 대표 4정책 D-30~D+29",
               font=dict(color=C['text'], size=16)),
    xaxis_title="rel_day (정책 발표일=0)", yaxis_title="PTEI total",
    paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
    font=dict(color=C['text']),
    legend=dict(bgcolor=PLOTLY_PANEL, bordercolor=C['grid'], font=dict(color=C['text'])),
    height=600, margin=dict(l=70, r=40, t=80, b=60),
    hovermode='x unified',
)
fig.update_xaxes(gridcolor=C['grid'], zeroline=True, zerolinecolor=C['accent5'], color=C['subtext'])
fig.update_yaxes(gridcolor=C['grid'], color=C['subtext'])

fig.write_html('figures/F13_daily_ptei_timeseries.html', include_plotlyjs='cdn')
print("✓ F13 (Plotly HTML)")

# F13 PNG 백업
fig_mpl, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
setup_dark(ax, "Figure 13 · 일별 PTEI 시계열 (대표 4정책)")
series_colors = [C['accent1'], C['accent2'], C['accent4'], C['accent3']]
for i, (pn, name) in enumerate(target.items()):
    sub = [r for r in panel if str(r[pn_col]) == str(pn)]
    sub.sort(key=lambda r: int(r[rd_col]))
    xs = [int(r[rd_col]) for r in sub]
    ys_p = [float(r[ptei_col] or 0) for r in sub]
    ax.plot(xs, ys_p, marker='o', markersize=4, linewidth=2,
            color=series_colors[i], label=f'pn={pn} {name}')
ax.axvline(0, color=C['accent5'], linestyle='--', alpha=0.5)
ax.set_xlabel("rel_day", color=C['text'])
ax.set_ylabel("PTEI total", color=C['text'])
ax.tick_params(colors=C['subtext'])
ax.legend(facecolor=C['panel'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=10)
ax.grid(color=C['grid'], linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/F13_daily_ptei_timeseries.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F13_daily_ptei_timeseries.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()

# ========================================
# F14 — Relevance gate level 분포 도넛
# ========================================
fig, ax = plt.subplots(figsize=(12, 7), facecolor=C['bg'])
setup_dark(ax, "Figure 14 · Relevance gate level 분포 — 99,539건 기사")

labels = [g[0] for g in GATE_LEVELS]
sizes = [g[1] for g in GATE_LEVELS]
colors_g = [C['accent4'], C['accent1'], C['accent2'], C['accent3'], C['accent5'], '#f0bb50']

wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors_g,
    autopct=lambda p: f'{p:.1f}%\n({int(p*sum(sizes)/100):,})',
    startangle=90, wedgeprops=dict(width=0.45, edgecolor=C['bg'], linewidth=2),
    textprops=dict(color=C['text'], fontsize=10), pctdistance=0.78,
)
for t in autotexts:
    t.set_fontsize(9)

# 중앙 텍스트
ax.text(0, 0.1, f'{sum(sizes):,}', ha='center', color=C['text'],
        fontsize=20, fontweight='bold')
ax.text(0, -0.2, '총 기사', ha='center', color=C['subtext'], fontsize=11)

plt.tight_layout()
plt.savefig('figures/F14_gate_levels.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F14_gate_levels.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F14")

# ========================================
# F15 — Plotly 인터랙티브 진행 상태 (Kanban)
# ========================================
done = [
    '방법론 v2.11 (60KB·15 섹션)',
    '정책 카드 v1.1 (50건)',
    '가설 자동 생성 (50×3)',
    '채널 시드 v0.2 (6채널)',
    'Dev 100 사람-사람 κ 0.95+',
    'Zero-shot baseline 측정',
    'Sanity 시리즈 4종',
    '99,539건 1차 PTEI 추론',
    '인용 논문 19편',
]
in_progress = [
    '보고서·시각화 작성',
]
waiting = [
    '서강대 CAR 결합 (H1~H5)',
    'Train 500 라벨링',
    'KLUE-RoBERTa fine-tune',
    '2차 PTEI 재추론',
    '5부작 기사 + 시각화',
]

fig = go.Figure()
cats = ['완료 (9)', '진행 중 (1)', '대기 (5)']
data = [done, in_progress, waiting]
col_colors = [C['accent1'], C['accent2'], C['accent4']]

for ci, (cat, items, color) in enumerate(zip(cats, data, col_colors)):
    n = len(items)
    fig.add_trace(go.Bar(
        x=[cat]*n, y=[1]*n,
        text=items, textposition='inside',
        marker=dict(color=color, line=dict(color=C['grid'], width=1)),
        hovertext=items, hoverinfo='text',
        textfont=dict(color=C['text'] if ci==2 else '#1a1d24', size=11),
        showlegend=False,
    ))

fig.update_layout(
    title=dict(text="Figure 15 · NLP 트랙 진행 상태 — Kanban",
               font=dict(color=C['text'], size=16)),
    barmode='stack',
    paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
    font=dict(color=C['text']),
    height=650, margin=dict(l=40, r=40, t=80, b=40),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    xaxis=dict(color=C['text'], gridcolor=C['grid']),
)

fig.write_html('figures/F15_progress_kanban.html', include_plotlyjs='cdn')
print("✓ F15 (Plotly HTML)")

# F15 PNG
fig_mpl, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
setup_dark(ax, "Figure 15 · NLP 트랙 진행 상태 — Kanban (완료 9 · 진행 1 · 대기 5)")
ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis('off')

cols = [(0.5, '완료', done, C['accent1']),
        (4.5, '진행 중', in_progress, C['accent2']),
        (8.5, '대기', waiting, C['accent4'])]
for cx, title, items, color in cols:
    # 헤더
    box = FancyBboxPatch((cx, 9.5), 3.5, 0.8, boxstyle="round,pad=0.1,rounding_size=0.2",
                          facecolor=C['panel'], edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(cx+1.75, 9.9, f'{title} ({len(items)})', ha='center', color=color,
            fontsize=12, fontweight='bold')
    # 항목
    for i, item in enumerate(items):
        y = 8.7 - i*0.85
        box = FancyBboxPatch((cx, y), 3.5, 0.7, boxstyle="round,pad=0.05,rounding_size=0.1",
                              facecolor=C['panel'], edgecolor=C['grid'], linewidth=1)
        ax.add_patch(box)
        ax.text(cx+1.75, y+0.35, item, ha='center', va='center',
                color=C['text'], fontsize=9, wrap=True)

plt.tight_layout()
plt.savefig('figures/F15_progress_kanban.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F15_progress_kanban.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()

# ========================================
# F16 — 자산 가치 평가 4분면 매트릭스
# ========================================
fig, ax = plt.subplots(figsize=(14, 9), facecolor=C['bg'])
setup_dark(ax, "Figure 16 · 자산 가치 평가 — 학술 × 실용 4분면")
ax.set_xlim(-1, 11); ax.set_ylim(-1, 11); ax.axis('off')

# 사분면 축
ax.axhline(5, color=C['grid'], linewidth=1.5)
ax.axvline(5, color=C['grid'], linewidth=1.5)
ax.annotate('실용 가치 →', xy=(10.5, 5), xytext=(10.5, 5),
            color=C['subtext'], fontsize=11, ha='right')
ax.annotate('학술 가치 ↑', xy=(5, 10.5), xytext=(5, 10.5),
            color=C['subtext'], fontsize=11, va='top', rotation=90)

# 4분면 라벨
ax.text(2.5, 9.5, 'High 학술 · Low 실용', ha='center', color=C['subtext'], fontsize=10, style='italic')
ax.text(7.5, 9.5, 'High 학술 · High 실용\n(견고한 자산)', ha='center', color=C['accent1'],
        fontsize=10, style='italic', fontweight='bold')
ax.text(2.5, 0.5, 'Low / Low\n(약한 자산)', ha='center', color=C['accent4'], fontsize=10, style='italic')
ax.text(7.5, 0.5, 'Low 학술 · High 실용', ha='center', color=C['subtext'], fontsize=10, style='italic')

# 자산 점 (학술 가치 x, 실용 가치 y)
assets = [
    ('방향 고정 설명모델 프레임', 8.5, 9, C['accent1'], 28),
    ('Dev 100 사람-사람 κ 0.95+', 8, 8.5, C['accent1'], 25),
    ('방법론 문서 (19편 매핑)', 7.5, 8, C['accent1'], 22),
    ('PTEI 식 (음수봉쇄·indicator gate)', 8.5, 7.5, C['accent1'], 22),
    ('공매도 금지 C5/C6 분리 사례', 7, 8.5, C['accent1'], 20),
    ('정책 카드 50건 · 8 subtype', 7, 7.5, C['accent1'], 20),
    ('Sanity 시리즈 4종', 6.5, 7, C['accent1'], 18),
    ('1차 전수 추론 (99K)', 5.5, 7.5, C['accent2'], 22),
    ('Zero-shot 모델 stance', 3.5, 4, C['accent4'], 20),
    ('Dev 100 단독 fine-tune', 3.5, 4.5, C['accent4'], 18),
    ('BigKinds 200자 본문', 4, 5, C['accent2'], 16),
    ('C4 dominance 잔존', 4, 4, C['accent4'], 16),
]

for name, x, y, color, size in assets:
    ax.scatter(x, y, s=size**2, c=color, alpha=0.75, edgecolors=C['bg'], linewidth=2, zorder=3)
    # 라벨 위치 조정
    offset_y = 0.35 if y < 5.5 else -0.35
    va = 'top' if y < 5.5 else 'bottom'
    ax.text(x, y+offset_y, name, ha='center', va=va, color=C['text'], fontsize=9,
            fontweight='bold' if color == C['accent1'] else 'normal')

# 주석
ax.text(5, -0.7, '강한 자산 (High/High) 8개 · 보강 필요 자산 (Low/Mid) 4개 — 종합 의미 약 70%',
        ha='center', color=C['subtext'], fontsize=11, style='italic')

plt.tight_layout()
plt.savefig('figures/F16_asset_quadrant.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.savefig('figures/F16_asset_quadrant.webp', dpi=150, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("✓ F16")

print("\nF11~F16 완료 (총 16개 figure)")
