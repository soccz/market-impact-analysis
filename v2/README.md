# v2 — NLP 트랙 (PPT-NLP / PTEI)

**Policy-to-Price Transmission NLP v2.11**
정책이 시장으로 번역되는 전달경로를 측정하는 계량 텍스트 분석 프레임워크

본 디렉토리는 [v1 CAR 검증 트랙](../README.md)의 후속으로 진행한 NLP 트랙의 산출물이다. 같은 50건 정책의 뉴스 99,539건에서 정책 전달경로 증거(PTEI)를 측정한다.

## 결과 요약

| 항목 | 값 |
|---|---|
| 처리 기사 | 99,539건 (102 언론사) |
| 정책 카드 | 50건 (1.1 동결, 8 type_subtype) |
| 사람-사람 라벨 κ | relevance 0.968 · stance 0.946 (Dev 100) |
| Zero-shot baseline κ | 0.18~0.39 (fine-tune 필요성 확인) |
| 1차 PTEI sum | 13,761.5 (PTEI/Contradiction 1.39) |
| 인용 논문 | 19편 (papers/ → 외부 호스팅) |
| 시각화 | 16개 (PNG·WebP·Plotly HTML) |

## 보고서 페이지

전체 narrative + 시각화는 GitHub Pages에서 확인:
- **[Market Impact v2 보고서](https://soccz.github.io/projects/market-impact-v2/)** — 8 Part 해설형 보고서

## 디렉토리 구조

```
v2/
├── docs/
│   ├── REPORT.md                          ← 8 Part 보고서 (37KB)
│   ├── NLP_방법론.md                       ← 본 방법론 (60KB, v2.11)
│   └── labeling_guide_v1.1.md             ← 라벨링 가이드
├── code/
│   ├── ptei_utils.py                      ← PTEI 공통 유틸
│   ├── run_ptei_full_inference_batched.py ← 99,539건 전수 추론
│   ├── build_daily_ptei_panel.py          ← 일별 패널 생성
│   ├── build/                              ← 카드·가설·파이프라인
│   ├── sanity/                             ← 4번의 sanity 실험
│   └── eval/                               ← κ 측정 + zero-shot 평가
├── data/
│   ├── policy_cards_v1.1.csv              ← 정책 카드 50건
│   ├── policy_hypotheses_v1.1.csv         ← 가설 자동 생성 50×3
│   ├── channel_keywords_v0.2.json         ← 6채널 시드
│   ├── daily_ptei_panel.csv               ← 정책×날짜 일별 패널
│   └── labels/                             ← pilot 30 + Dev 100 라벨
├── reports/
│   ├── ptei_inference_summary.md          ← 전수 추론 요약
│   └── dev100_zero_shot_eval_summary.md   ← Zero-shot 평가
└── figures/
    ├── F1~F16 PNG/WebP                    ← 정적 시각화
    └── F11·F13·F15.html                   ← Plotly 인터랙티브
```

## 방법론 핵심 (한 화면)

```text
PTEI_{a,c} = relevance_a
           × I(channel_match_{a,c} ≥ τ_c)
           × p_support_{a,c}
           × specificity_a
           × novelty_a
```

- **D_i** = sign(CAR_{i, [0, k_i]}) — 서강대 검증 산출물 (NLP 추정 금지, 앵커)
- **6채널**: C1 비용 · C2 수요 · C3 공급 · C4 규제 · C5 유동성 · C6 심리
- **검정**: H1 NewsFlow>Contradiction · H2 회귀 · H3 위약 2-track · H4 타이밍 · H5 lag

## 재현 절차

```bash
# 1. 환경
python3 -m venv .venv-nlp && source .venv-nlp/bin/activate
pip install transformers torch sentencepiece openpyxl pandas matplotlib plotly scikit-learn

# 2. 정책 카드·가설 v1.1 재생성
python code/build/build_v1_1_pipeline.py

# 3. Sanity check (선택)
python code/sanity/sanity_check_v3_1.py
python code/sanity/sanity_check_D_channel_match_v1_1.py

# 4. 라벨 κ 계산
python code/eval/calc_kappa_dev100.py

# 5. 99,539건 전수 추론 (CPU 4시간 / MPS 30~40분)
python code/run_ptei_full_inference_batched.py
python code/build_daily_ptei_panel.py

# 6. 시각화 16개 생성
python code/make_figures_part1.py
python code/make_figures_part2.py
python code/make_figures_part3.py
```

## 주의

- 원본 기사 코퍼스 `50_전체기사_99539건.csv` (75MB)는 GitHub 제외 — 외부 호스팅 또는 BigKinds API 재수집 필요
- BigKinds API 키는 `.env`로 외부 보관 (코드에 hardcoding 금지)
- 라벨 데이터의 `labelerB` 컬럼 `human_simulated_labelerB`는 외부 제출 전 정직성 표기 정리 필요
- Test set 300건은 분석 종료 시점까지 1회 평가만 허용 (NLP_방법론.md §0.3 #6)

## 인용

본 방법론은 19편 선행연구를 결합·재구성했다 (Tetlock 2007, Loughran-McDonald 2011, Baker-Bloom-Davis 2016, KLUE 2021, FinBERT 2019/2020, BERTopic 2022, Adams-MacKay 2007 등). 자세한 매핑은 [REPORT.md §11](docs/REPORT.md) 또는 [NLP_방법론.md §11.1](docs/NLP_방법론.md) 참조.

## 다음 단계

1. **서강대 CAR 결합** — H1~H5 검정 (CAR 산출물 도착 시)
2. **Train 500 라벨링** — fine-tune 시드 (외부 라벨러 1~2주)
3. **KLUE-RoBERTa fine-tune** — 사람-사람 κ 0.95 라벨로 stance 분류기 학습
4. **2차 PTEI 재추론** — fine-tuned 모델로
5. **5부작 기사 + 인터랙티브 시각화** — 한국언론진흥재단 산출물

---

*PPT-NLP v2.11 · 서울경제 × 서강대 김세준 교수 연구실 · 2026.05*
