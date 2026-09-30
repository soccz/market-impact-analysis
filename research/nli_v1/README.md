# Korean policy NLI — v1 aggregate record

2026-09-30에 실행한 3 seed 학습 비교와 사후 원인 진단의 집계·설계·해시입니다. [전체 설명](../../docs/KOREAN_POLICY_NLP.md)을 참고하세요.

`analysis.json`, `protocol.json`, `followup_analysis.json`, CSV 및 잠금 파일은 로컬 원 결과를 그대로 복사했습니다. `environment.json`의 `local only`는 원 실험 기록 당시 상태입니다. `research_record.json`에는 공개 집계로 내보낸 상태 설명을 추가했습니다.

전체 모델 가중치·원시 예측·KLUE 원본·공식 페이지 캐시는 여기 포함하지 않았습니다. 원 기록에 적힌 해시는 로컬 보존 파일을 식별하며, 이 폴더만으로 전체 학습을 재현하거나 모든 잠금을 검증할 수 있다는 뜻이 아닙니다. 공개 집계 수치와 후속 v2의 재실행 가능한 실험을 구분합니다.

KLUE는 [공식 저장소](https://github.com/KLUE-benchmark/KLUE)의 CC-BY-SA 4.0 자료를 사용했습니다. 사전학습 모델은 [KLUE-RoBERTa](https://huggingface.co/klue/roberta-base)입니다. 합성 정답은 AI가 구성한 규칙이며, 공식 문서 18주장도 독립 사람 판독 정답집이 아닙니다. 고객 데이터·납품 구현을 사용하지 않은 개인 후속 실험입니다.
