# Dev 100 Zero-shot Baseline Evaluation

- model: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- consensus agreed rows: `95`
- excluded labeler disagreements: `5`
- evaluated rows in this run: `100`

| target | n | accuracy | macro-F1 | kappa |
|---|---:|---:|---:|---:|
| relevance 3-class | 95 | 0.589 | 0.475 | 0.386 |
| relevance binary rel>0 | 95 | 0.663 | 0.638 | 0.346 |
| stance all agreed | 95 | 0.589 | 0.579 | 0.325 |
| stance consensus rel>0 | 75 | 0.480 | 0.484 | 0.177 |
