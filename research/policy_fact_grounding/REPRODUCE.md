# 재현

Python 3.12.9를 사용했다. 저장소 루트에서 실행한다. 원시 캐시 없이도 공개 구조와 점수를 재계산할 수 있다.

```bash
python -m pip install -r research/policy_fact_grounding/requirements.txt
python research/policy_fact_grounding/verify.py
```

검증은 17개 행동 검사(630개 금액 표기 산술 대조 포함), 동결 파일, 이전 10패키지 해시, 216개 새 응답의 모델·시간 메타데이터, 5개 평가 묶음, 반례 20개를 재확인한다. `research_record.json`은 현재 공개 파일의 로컬 무결성 기록이다. 금액 격자 검사는 방법 동결 후 추가한 산술 검증이며 새로운 언어모델 성능 평가가 아니다.

원본 두 파일을 별도 디렉터리에 받아 같은 모델 입력을 만든다. 이 명령은 다운로드 바이트와 전체 추출문, 선택 근거의 해시가 모두 같아야 성공한다. 공식 사이트의 파일이 교체되거나 접근이 막히면 임의로 다른 버전을 쓰지 않는다.

```bash
python research/policy_fact_grounding/prepare_inputs.py --sources /tmp/policy-grounding/sources --download --out /tmp/policy-grounding/official_inputs.json
```

모델·서비스 준비는 [앞선 재현 기록](../policy_compositional_reasoning/REPRODUCE.md)을 따른다. 신규 호출은 저장된 응답 재채점과 구별한다. 로컬 Ollama 0.33.2, 모델 해시는 이전 모델 기록과 같아야 한다.

```bash
python research/policy_fact_grounding/run_experiment.py --inputs research/policy_fact_grounding/metamorphic.json --cache /tmp/policy-grounding/metamorphic --dest /tmp/policy-grounding/recomputed-metamorphic
python research/policy_fact_grounding/run_experiment.py --inputs /tmp/policy-grounding/official_inputs.json --cache /tmp/policy-grounding/official --dest /tmp/policy-grounding/recomputed-official
python research/policy_fact_grounding/verify.py --cache /tmp/policy-grounding
```

마지막 검증 명령은 **공개한 바로 그 원시 캐시**와 비교하는 용도다. 새로 추론한 응답은 하드웨어·서비스·시각 때문에 기존 해시와 같다고 보장하지 않으므로 별도 실험으로 기록한다. 원시 캐시는 공개 저장소에 포함되지 않는다. 공개 검증만으로 비공개 원시 산문을 독립 확인했다고 주장할 수 없다.

`rescore_previous.py`는 기존 80주장·두 모델의 같은 출력을 다시 채점한다. `counterexamples.py`는 기존 규칙 반례를 만들며 `witness_audit.py`는 별도 계산으로 검산한다. 이 둘에는 모델 호출이 없다. 그림은 `plot_results.py`로 공개 요약에서 다시 그린다. 동결된 코드·주장·참조 파일을 수정하면 최초 결과의 동결 검증이 실패하므로 후속 방법은 별도 파일과 버전으로 기록해야 한다.
