# 재현과 실행 범위

Python 3.12에서 저장소 루트를 기준으로 실행합니다. 저장된 결과의 재생에는 모델 호출이 필요하지 않습니다.

```bash
python -m unittest discover -s research/policy_document_closure -p 'test_*.py'
python research/policy_document_closure/verify.py
python research/policy_document_closure/plot_results.py
```

첫 명령은 조건 계산·누락 사실의 상관 보존·순환 참조 거부·입력 분리·독립 Python 술어를 검사합니다. 둘째는 120개 공개 정제 응답으로 사례별 점수와 의미 변화 쌍, 추출 그래프의 동치 검사를 재생하고, 동결 및 과거 연구 산출물의 해시를 확인합니다. 과거 검증 스크립트 일부는 당시 패키지 총수를 고정했으므로 새로운 연구가 추가된 현재 checkout에서는 이 최종 검증기를 사용합니다. 과거 버전의 그대로인 검증은 해당 과거 커밋에서 실행할 수 있습니다.

공식 HTML 전체 본문과 원시 모델 응답은 재배포하지 않습니다. 두 공식 출처의 본문은 다음과 같이 로컬에 복원할 수 있습니다. TLS 검증을 유지하며, 본문이 수집 당시와 달라지면 원문 해시 검사에서 멈춥니다. 다운로드 성공이 원문 버전 일치를 의미하지 않습니다.

```bash
python research/policy_document_closure/rehydrate.py --out /tmp/policy-document-inputs --download
```

`#contents` 안의 텍스트 전체를 읽고 script/style만 제외합니다. 최대 420자 기준으로 줄을 묶으며, 하나의 원래 줄을 잘라 숨기지 않습니다. 연결된 모든 법령·첨부파일을 수집한 것은 아닙니다. 질문 범위와 필드는 사람이 정했고, 개별 필요한 조항을 미리 골라 모델에 넘기지는 않습니다. 원문으로부터 참조 근거 문단을 별도로 지정한 것은 검색 회수율의 평가용이며 요청에 넣지 않습니다.

로컬 Ollama에 기록된 두 모델이 준비되어 있다면 다음 명령으로 다시 추론할 수 있습니다. 재실행은 새로운 측정이며 원래 응답과 동일한 출력을 보장하지 않습니다. 새로운 캐시 디렉터리를 사용하고 공개된 결과를 덮어쓰지 마세요.

```bash
python research/policy_document_closure/experiment.py --inputs /tmp/policy-document-inputs/inputs.json --cache /tmp/policy-document-new-run
```

모델은 `qwen3.5:9b`, `policy-kanana15-public:q4km-f7ae0cc1`이며 실제 digest는 `results/predictions.json`에 있습니다. 온도 0, seed 20261002, 문맥 16384, 직접 판독 출력 예산 256, 그래프 추출 4096입니다. Qwen은 `think=false`로 실행했습니다. 추가 학습·예측 후 답변 피드백·성능 개선을 위한 조건 그래프 재시도는 없습니다. Kanana 첫 요청은 CPU 실행에서 600초 전송 제한에 걸려 저장되지 않았습니다. 별도 GPU 서버에서 같은 요청을 다시 실행한 전송 재시도 1회는 [실행 사고 기록](runtime_incident.json)에 따로 남겼습니다. `run_endpoint.py --base http://127.0.0.1:11436`는 주소만 바꾸며 요청 본문·동결 파일을 수정하지 않습니다. 모델 로딩만 한 요청 1회와 미저장 요청은 새 저장 응답 120개에 포함하지 않습니다. 소스·참조·실험 코드 고정은 로컬 해시 스냅샷이며 외부 사전등록은 아닙니다.

보관한 비공개 원본이 있는 환경의 전체 검증 명령은 다음과 같습니다. private root에는 `inputs.json`, 두 HTML 파일, `cache/`가 있어야 합니다.

```bash
python research/policy_document_closure/verify.py --private /path/to/private-root
```

이 검사는 요청·입력·원시 응답·모델 digest·동결 이후 시각·출력 종료 상태까지 대조합니다. 공개 점수 재생은 가능한 범위에서 검증하고, 비공개 원시 응답 없이 원래 모델 호출 자체를 독립적으로 증명한다고 주장하지 않습니다.
