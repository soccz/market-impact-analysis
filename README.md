# Policy & Market Impact Analysis

정책 변수와 자본시장 반응에 관한 정량적 분석 코드입니다.

## Structure

```
.
├── verify_50_events.py              # 50건 이벤트 주가 변동 검증
├── scripts/
│   ├── build_reproducible_data_package.py   # 검증 가능 데이터 패키지 생성
│   ├── collect_japan_regulation_dataset.py   # 사례 B: 소재 공급 충격 분석
│   └── collect_kchips_dataset.py            # 사례 A: 반도체 세제 정책 분석
├── data/
│   └── verification_results.csv     # 50건 검증 결과
└── assets/
    ├── case_a_visualization.png     # 사례 A 시각화
    └── case_b_visualization.png     # 사례 B 시각화
```

## Overview

- **50건 이벤트 검증**: 1998~2025년, 5개 산업 섹터 x 10건
- **이벤트 스터디**: CAR/AR 산출, KOSPI 벤치마크 대비 초과수익 측정
- **NLP 감성분석**: 뉴스 기사 감성(긍정/중립/부정) 분류 및 순감성지수 추적
- **인과관계 검정**: Granger causality, ADF 단위근 검정, 비율검정

## Requirements

```
pip install FinanceDataReader pandas numpy matplotlib scipy statsmodels openpyxl beautifulsoup4 requests
```

## Usage

```bash
# 50건 이벤트 주가 변동 검증
python verify_50_events.py

# 사례 분석 데이터셋 수집 (API 키 필요)
export BIGKINDS_KEY="your_api_key"
python scripts/collect_kchips_dataset.py
python scripts/collect_japan_regulation_dataset.py

# 검증 가능 데이터 패키지 생성
python scripts/build_reproducible_data_package.py
```

## Verification

| 검증 항목 | 결과 |
|----------|------|
| FDR 수익률 재현 (2014~) | 33/33 PASS, 오차 0.00% |
| 크롤링 데이터 재현 (~2013) | 17/17 PASS, 오차 0.00% |
| 방향 일치 검증 | 50/50 PASS |
| 날짜 팩트체크 | 41 PASS / 9 WARN / 0 FAIL |

## Project Page

[https://soccz.github.io](https://soccz.github.io)
