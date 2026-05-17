# PTEI Hybrid Inference Summary

- input: `50_전체기사_99539건.csv`
- article scores: `data/processed/article_scores_hybrid.csv`
- daily panel: `data/processed/daily_ptei_panel.csv`
- model: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- gate: relevance v3.1 + channel_match hybrid, MPS batch run

## Counts

- articles: `99539`
- policies: `50`
- daily panel rows: `2454`
- relevance >= 0.4: `53568`
- active channel articles: `28603`
- nonzero PTEI articles: `28603`
- PTEI total sum: `13761.546`
- Contradiction total sum: `9925.894`

## Predicted Stance

- neutral: `76342`
- support: `16445`
- contradict: `6752`

## Relevance Gate Levels

- NO_MATCH: `45971`
- L1_primary_exact: `32975`
- L2_primary_token_AND: `15974`
- L1.5_secondary_exact: `2134`
- L3_alias: `1456`
- L2.5_secondary_token_AND: `1029`

## Channel Totals

| channel | gate count | PTEI sum | contradiction sum |
|---|---:|---:|---:|
| C1 | 3647 | 1422.730 | 1011.747 |
| C2 | 4744 | 1678.420 | 1332.175 |
| C3 | 7090 | 2585.142 | 1761.073 |
| C4 | 12793 | 4152.368 | 3369.036 |
| C5 | 4900 | 1760.089 | 1227.142 |
| C6 | 5208 | 2162.797 | 1224.721 |

## Top 10 Policies by PTEI Sum

| pn | alias | articles | active | PTEI sum | contradiction sum |
|---:|---|---:|---:|---:|---:|
| 3 | 일본 수출규제 | 5000 | 3618 | 1884.755 | 1812.970 |
| 16 | 공매도 금지 4차 | 3231 | 2340 | 1324.621 | 930.700 |
| 28 | 8·2 대책 | 5000 | 2100 | 1221.949 | 842.131 |
| 17 | 100조 패키지 | 5000 | 1511 | 829.040 | 513.774 |
| 26 | 4·1 대책 | 4776 | 1557 | 708.224 | 396.285 |
| 31 | 의약분업 | 4804 | 1376 | 619.723 | 471.173 |
| 29 | 9·13 대책 | 3443 | 1443 | 564.862 | 540.303 |
| 23 | 8·31 대책 | 3952 | 1358 | 541.264 | 455.676 |
| 20 | 자본시장법 | 2229 | 751 | 405.385 | 211.139 |
| 50 | 그린뉴딜 | 5000 | 648 | 360.928 | 155.779 |
