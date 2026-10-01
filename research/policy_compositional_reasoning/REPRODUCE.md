# 실행표와 재사용 규칙

공개 결과의 재채점은 `verify.py` 하나로 실행한다. 아래는 동일 설계를 새로 추론하는 절차다. 모델 설치와 Ollama 실행은 별도 필요하며 모델 태그뿐 아니라 `models.json`의 digest를 확인한다. 원시 캐시는 저장소 밖에 두고 기존 동결 결과를 덮어쓰지 않는다. 새로운 실행의 시각은 기존 실험과 다르므로 이번 기록과 같은 실험으로 취급하지 않는다.

## runner 대응

| 결과 디렉터리 | 입력 | runner | 채점 |
|---|---|---|---|
| development | development.json | infer.py | evaluate.py --method baseline |
| development_explicit | development.json | infer_explicit.py | evaluate.py --method explicit |
| development_examples | development.json | infer_logic_examples.py | evaluate_variant.py |
| official | 재생성 official_inputs.json | infer_explicit.py | evaluate.py --method explicit |
| official_examples | official_inputs.json | infer_logic_examples.py | evaluate_variant.py |
| composition | composition.json | infer_explicit.py | evaluate.py --method explicit |
| composition_examples | composition.json | infer_logic_examples.py | evaluate_variant.py |
| capability_base | 재생성 capability_inputs.json | infer_logic_examples.py | evaluate_capability.py --method base |
| capability_extended | capability_inputs.json | infer_extended.py | evaluate_capability.py --method extended |

모든 runner는 `--inputs INPUT --out CACHE/SPLIT/MODEL --model MODEL`을 받는다. 모델 식별자는 `qwen`, `kanana-public`이다. 채점기는 `--inputs INPUT --cache CACHE/SPLIT --dest OUTPUT`을 받는다. 원본 결과와 분리된 OUTPUT에 쓴다. 두 모델을 모두 실행한 뒤 채점한다.

```bash
python research/policy_compositional_reasoning/infer_explicit.py --inputs ../private/official_inputs.json --out ../private/fresh/official/qwen --model qwen
python research/policy_compositional_reasoning/infer_explicit.py --inputs ../private/official_inputs.json --out ../private/fresh/official/kanana-public --model kanana-public
python research/policy_compositional_reasoning/evaluate.py --inputs ../private/official_inputs.json --cache ../private/fresh/official --dest ../private/reports/official --method explicit
```

개발 지시 명확화는 최초 개발의 direct만 재사용했다. 논리 예시 실험은 대응하는 지시 명확화 실험의 direct/profile을 재사용했다. capability_extended는 capability_base의 direct만 재사용했다. 해당 하위 디렉터리를 새 캐시에 복사하면 runner가 요청 해시와 모델 digest가 같은 첫 응답을 확인하고 건너뛴다. 재사용 없이 모두 새로 실행하면 원래 연구보다 호출 수와 변동 요인이 늘어난다.

## 검증 피드백 재판독

`repair.py --select`는 저장한 공개 예측을 기준으로 선택 파일을 생성하므로 동결된 공개 패키지에서는 다시 실행하지 않는다. 원래 실행에서 선택한 10개와 이유는 `repair_selection.json`에 보존했다. `repair.py --cache CACHE --split official_repair --inputs INPUT`은 official에서, composition_repair는 composition에서, capability_repair는 capability_extended에서 시작한다. 한 번의 규칙 재추론만 수행하고 direct/profile 및 미선택 규칙은 복사한다.

이 runner는 현재 패키지의 `results/SPLIT`에 보고서를 쓰므로 **새 추론 연구는 별도 체크아웃**에서 실행한다. 고정 첫 응답의 재현 검증에는 모델 호출 없이 `verify.py --cache CACHE`를 사용한다. 재판독 요청은 기존 공개 규칙 출력과 검증 피드백으로 다시 만들며 원문·질문·참조 답의 입력 분리를 검사한다.

## 공개 재현의 범위

정제된 공개 예측으로 점수·혼동행렬·규칙 동등성·필요 사실·호출 중복을 계산한다. 공개 원문을 다운로드해 선택 입력도 재생성할 수 있다. 원시 모델 응답 전체와 자유서술은 재배포하지 않으므로 원시 응답 해시 자체의 대조는 그 캐시를 가진 환경에서만 가능하다. 로컬 기록은 이 한계를 숨기지 않는다.
