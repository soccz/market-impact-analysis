# 재현

저장소 루트에서 Python 3.12와 [requirements.txt](requirements.txt)의 의존성을 사용합니다. 모델은 기존 연구와 같은 Qwen3.5:9b 및 공개 Kanana 양자화 프로필, Ollama 0.33.2입니다. 모델 이름·digest·시각·요청/응답 해시는 각 `predictions.json`에 있습니다. 온도 0, seed 20261001, 문맥 8192이며 요청별 출력 예산은 코드에 보존했습니다. 학습은 실행하지 않았습니다.

```bash
python -m unittest discover -s research/policy_source_alignment -p 'test_*.py'
python research/policy_source_alignment/verify.py
python research/policy_source_alignment/plot_results.py
```

첫 두 명령은 모델을 다시 호출하지 않습니다. 공개 정제 응답·규칙·사실로 점수와 갱신을 재생하고, 독립 Fraction 계산으로 7개 피드백 반례를 검사합니다. 47개 행동 테스트에는 448개 독립 연령 경계 조합이 포함됩니다. 이전 연구 2,354개 산출물도 해시로 확인합니다. 모델 실행이 같다고 원문 주석의 타당성까지 입증하는 것은 아닙니다.

원문과 원시 응답을 보관한 별도 캐시가 있으면 다음과 같이 요청·입력·응답·출처까지 확인합니다. `CACHE`는 사용자가 보관한 비공개 캐시 경로입니다. 인접 디렉터리에 이전 `policy_fact_grounding`, `policy_selective_update` 캐시가 필요합니다.

```bash
python research/policy_source_alignment/verify.py --cache "$CACHE"
```

원문 출처는 [sources.json](sources.json), [confirmation_sources.json](confirmation_sources.json)에 있습니다. 공식 HTML을 TLS 검증으로 내려받고 BeautifulSoup으로 script/style을 제거한 뒤 `get_text('\n', strip=True)`로 읽었습니다. 수집 당시 원본·추출문 해시를 검사해야 합니다. 변경되는 홈페이지의 현재 바이트가 수집 당시 바이트와 같다고 가정하지 않습니다. [prepare_transfer.py](prepare_transfer.py)는 캐시의 추출문과 동결된 위치/해시로 첫 새 문서 입력을 복원합니다. 확인 자료도 `confirmation_reference.json`의 동일한 위치 구조로 복원합니다.

모델 재실행은 저장 응답과 같은 결과를 보장하지 않으며 기존 파일을 덮어쓰면 안 됩니다. 빈 캐시를 사용하십시오. `source_method.py`는 최초 고정 방법이고 `age_extension.request`는 사후 연령 확장입니다. 후자는 기존 `bridge.infer.request`에 주입해 동일 runner로 실행합니다. 표본·질문·참조를 모델 요청에 함께 전달하지 않으며 프로필 요청에는 정책 원문을 넣지 않습니다.

개발 단계의 최초·명확화·반례 재시도·ALL/ANY 출력은 각각 `results/development`, `clarification`, `repair`, `explicit_logic`에 있습니다. `aligned`는 새 모델 호출 없이 사실을 보정한 결과입니다. `transfer`는 창원·수원 최초 결과, `transfer_repair`는 같은 문서의 연령 보완 재검사와 자유 추출 비교군, `confirmation`은 고정 후 국민연금공단 확인입니다. `revision`은 이전 전후 문서를 재사용한 갱신 통합입니다. 전부 합쳐 **335개 고유 저장 응답**이며 재사용된 레코드를 추가 호출로 세지 않습니다.

`research_record.json`은 공개 파일의 해시 목록입니다. 그림은 저장 집계에서 생성되며 PDF와 PNG를 함께 제공합니다. HTML 페이지의 상호작용은 저장 결과의 탐색이고 브라우저 내 실시간 추론이 아닙니다.
