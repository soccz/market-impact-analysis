# 2차 PTEI 신호 분석 — 핵심 가설 검증

## 본 보고서가 검증하는 것
> **신문기사의 영향력으로 정책 방향성에 맞는 주가 흐름이 보여진다** — 그 전제 조건인 "뉴스가 정책 방향과 일치하는가?"를 먼저 검증.

CAR(서강대) 데이터 도착 전, **뉴스 자체 시그널만으로** 검증 가능한 부분을 정리합니다.

---

## 핵심 결과 한눈에

### 1. 정책 50건 중 뉴스 방향 일치
- **전체 일치율**: 33/50 정책 (66.0%)
- 호재 정책(+): 33/33 (100.0%) 일치
- 악재 정책(-): 0/17 (0.0%) 일치

### 2. 시점별 일치율
- **선반영 (D-7~D-1)**: 33/50 정책 (66.0%)
- **발표 (D0~D+1)**: 31/48 정책 (64.6%)
- **후폭풍 (D+2~D+5)**: 32/49 정책 (65.3%)


---

## 그림

1. [v2_fig1_ptei_timeline.png](../figures/v2_fig1_ptei_timeline.png) — 1차 vs 2차 PTEI 시점별 평균
2. [v2_fig2_heatmap.png](../figures/v2_fig2_heatmap.png) — 정책×시점 PTEI 매트릭스
3. [v2_fig3_alignment_per_policy.png](../figures/v2_fig3_alignment_per_policy.png) — 정책별 일치 점수
4. [v2_fig4_phase_alignment.png](../figures/v2_fig4_phase_alignment.png) — 시점별 일치율
5. [v2_fig5_direction_split.png](../figures/v2_fig5_direction_split.png) — 호재/악재 정책 시점별 신호

---

## 정책별 신호 (Top 5 일치 / Bottom 5)

### Top 5 — 뉴스가 정책 방향과 가장 잘 일치
| pn | alias | direction | signal_mean |
|---|---|---|---|
| 34 | 시밀러 가이드 | + | +0.966 |
| 48 | 첨단전략산업법 | + | +0.919 |
| 4 | 소부장 대책 | + | +0.873 |
| 40 | CDMO 특별법 | + | +0.866 |
| 6 | 반도체 초강대국 | + | +0.865 |

### Bottom 5 — 뉴스가 정책 방향과 가장 어긋남
| pn | alias | direction | signal_mean |
|---|---|---|---|
| 3 | 일본 수출규제 | - | -0.886 |
| 33 | 황우석 폭로 | - | -0.854 |
| 39 | 의대증원 | - | -0.844 |
| 11 | 5개 은행 퇴출 | - | -0.843 |
| 22 | 10·29 대책 | - | -0.841 |


---

## 해석 가이드

- **signal_mean > 0**: 뉴스가 정책 의도 방향을 그대로 보도 (호재 정책엔 호재 톤, 악재 정책엔 악재 톤)
- **signal_mean < 0**: 뉴스가 정책 의도와 반대로 보도 (호재 정책에 비판·우려 톤 등)
- **시점별 차이**: 선반영 단계에서 일치율이 높으면 = 뉴스가 정책 발표를 미리 예고
- **D0~D+1 일치율**: 가장 강해야 정상 (발표 직후 공식 보도)

---

## 데이터 출처

- 입력: `data/processed/article_scores_v2.csv` (fine-tuned KLUE-RoBERTa, 99,539건)
- 정책 방향: `policy_cards_v1.1.csv` (D_i_initial)
- 윈도우: rel_day ∈ [-7, 5]

## 다음 단계

서강대 CAR(누적초과수익률) 도착 시:
1. PTEI peak day vs CAR peak day 비교
2. PTEI × CAR cross-correlation (lag 분석)
3. 4유형 분류 (즉시·선반영·지연·설명보강)
