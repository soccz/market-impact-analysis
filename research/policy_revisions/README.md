# Korean policy revision probe — v2

soccz의 개인 후속 연구. AI 지원으로 작성한 코드·합성 자료·임시 라벨이며 독립 사람 판독은 아직 없습니다. [연구 과정·모든 결과·한계](../../docs/KOREAN_POLICY_NLP.md)를 먼저 읽어 주세요.

이 묶음은 고객 데이터와 납품 코드를 포함하지 않습니다. 정책의 금액 구절을 원문 위치에 연결하고, 정정·정책 변경·동일 금액·관계 불명을 분류하는 작은 실험입니다. 통화 파서는 아라비아 숫자로 쓴 원화만 다룹니다. 문서 수집, 모든 개체·날짜·시행 상태 추출, 법률 해석을 수행하지 않습니다.

## 재현

실행 환경: Python 3.12.9, NumPy 1.26.4, scikit-learn 1.7.2, PyTorch 2.2.2, transformers 4.39.3. `requirements.txt`는 관측한 핵심 버전입니다. 환경에 맞는 PyTorch 빌드는 별도로 준비할 수 있습니다. GPU가 필요하지 않습니다.

저장소 루트에서:

```bash
python research/policy_revisions/verify.py
python research/policy_revisions/run.py --out /tmp/policy-revisions-character
python research/policy_revisions/frozen_encoder.py \
  --encoder /path/to/klue-roberta-base-snapshot \
  --out /tmp/policy-revisions-encoder
python research/policy_revisions/matched_char.py --out /tmp/policy-revisions-matched
```

encoder는 `klue/roberta-base` revision `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da`의 로컬 디렉터리를 전달합니다. 자동 다운로드하지 않습니다. 모델·토크나이저는 해당 제공자의 이용 조건을 따릅니다. 원 공개 가중치를 배포하지 않으며, `results/encoder/*_probe.npz`는 이 실험에서 학습한 선형 분류기 계수입니다. 피클 파일을 실행할 필요가 없습니다.

`build_data.py`는 보존한 자료가 있으면 덮어쓰지 않습니다. 재생성하려면 새 작업 사본에서 빈 `data/`를 준비하고 실행합니다. 코드 포맷까지 포함해 잠금 해시에 반영돼 있으므로 첫 실행 코드는 자동 포맷하거나 수정하지 않았습니다. 새 방법은 새 파일·새 잠금으로 만듭니다.

## 파일과 상태

| 파일 | 의미 |
|---|---|
| `protocol.json`, `freeze.json` | 문자 모델의 첫 학습 전에 저장한 설계·입력·코드 해시 |
| `encoder_freeze.json` | 첫 결과를 본 뒤 고정한 문맥 표현 탐색; 독립 재검증 아님 |
| `matched_freeze.json` | 문맥 모델과 입력 정규화를 통일한 문자 비교; 사후 탐색 |
| `data/*.jsonl` | 합성 문장·라벨·원문 구절 위치와 짧은 공식 문서 사례 |
| `results/character/` | 처음 고정한 문자 기반 실험 |
| `results/encoder/` | 고정 한국어 사전학습 표현 + 선형 분류기 |
| `results/matched_character/` | 입력 통일 비교 |
| `verification.json` | 원시 결과에서 독립 재계산한 지표·유형별 답변 수 |
| `independent_review.csv` | 서로 다른 합성 표현 40개, 빈 판독 칸 |

지표의 `joint_exact`는 보류 전의 원예측이 관계와 모든 금액 역할·위치·정규값을 맞힌 비율입니다. `correctly_answered_fraction`은 전체 입력 중 보류하지 않고 올바르게 답한 비율입니다. `selective_joint_error`는 답한 입력 중 틀린 비율입니다. **보류를 정답으로 세지 않습니다.** `undetermined`는 의미 라벨이고 보류는 별도의 출력 상태입니다.

합성 평가 768개는 숫자·사업명을 가린 뒤 **40개 표현**입니다. 독립적인 768개 실제 정책 문장으로 해석하면 안 됩니다. 선택적 문맥 모델은 132개에 답했지만 정정·정책 변경에는 0개 답했습니다. 공식 5사례에서는 1개에 답했고 그 답이 틀렸습니다. 확신 점수가 실제 정확도 확률로 보정됐다고 주장하지 않습니다.

## 사람 판독 준비

판독자에게는 `independent_review.csv`와 아래 정의만 먼저 전달합니다. 이 저장소의 생성 정답과 모델 예측을 미리 읽은 판독은 블라인드 판독이 아닙니다.

- **correction:** 이전에 잘못 기재한 수치를 바로잡았다는 주장이 명시됨.
- **policy_change:** 오류 정정이 아니라 정책·재원·지원 규모를 바꿨다는 주장이 명시됨.
- **equivalent:** 다른 단위로 같은 금액을 적었다고 명시됨.
- **undetermined:** 위 관계가 확정되지 않음. 조건·부인·미확정·사유 누락을 메모에 구분.
- 이전/현재 금액 및 수정 사유를 원문 그대로 짧게 옮기고, 모호하면 근거 없이 채우지 않음.
- 계획·예정 상태의 해석에 이견이 있으면 별도로 남김. 2인 독립 판독 뒤 이견 조정은 후속 절차이며 아직 수행하지 않음.

공식 사례는 인천광역시·정책브리핑·원주시의 짧은 출처 표시 인용입니다. `data/official_cases.jsonl`에 URL·문서 시점·인용 위치·해시를 기록했습니다. 전체 페이지 캐시는 배포하지 않습니다. 웹 페이지가 바뀌면 재수집 값과 해시도 바뀔 수 있습니다.
