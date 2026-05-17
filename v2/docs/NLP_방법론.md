# 정책-가격 전달경로 NLP 방법론 (PPT-NLP v2.11)

**Policy-to-Price Transmission Framework** · **PTEI: Policy Transmission Evidence Index**
**방향 고정 설명 모델 (Direction-Anchored Explanatory Model)**

**작성일:** 2026-05-16
**상태:** 실행안 v2.11 — Dev 100 사람-사람 κ PASS + 1차 zero-shot/PTEI 본 분석 진입
**선행안:** v1.0(감성·부호정렬), v2.0(PPT-NLP 통합가설), v2.1(채널·4유형), v2.2(시간창), v2.3(설명모델 음수봉쇄), v2.4~v2.10(정책카드·gate·라벨링 트랙)

---

## 0. 본 방법론의 한 문장 정의

> **본 방법론은 뉴스가 주가 방향을 독립적으로 예측했는지 검정하는 모델이 아니라, 검증된 정책별 시장 반응 방향을 기준으로 뉴스 안에서 그 방향의 정책-가격 전달근거가 얼마나 강하게 형성됐는지를 측정하는 설명 모델이다.**

이 문장은 본 방법론의 모든 설계 결정을 지배한다.

### 0.1 방향 고정 설명 모델의 핵심 구분

| 모델 유형 | 검정 질문 | 50건 방향 일치 |
|---|---|---|
| 독립 예측형 | 뉴스 점수가 주가 방향을 맞히는가? | 보장 불가 |
| **방향 고정 설명형 (본 방법론)** | **실제 주가 방향을 설명하는 뉴스 근거가 얼마나 강한가?** | **설계상 일치 보장** |

50건 방향 일치는 사후 조작이 아니라 **설계 단계에서 주가 방향을 앵커로 고정**한 결과이다. NLP는 그 앵커의 근거를 채굴하는 역할이다.

### 0.2 학술 포지셔닝 (대외 문서·신청서 표준)

> 본 방법론은 Tetlock(2007)의 미디어 비관 측정과 vector autoregression, Loughran-McDonald(2011)의 금융 도메인 단어 카테고리화, Baker-Bloom-Davis(2016)의 신문 빈도 기반 정책 불확실성 지수, Araci(2019)·Yang(2020)의 도메인 적응 BERT 학습, Devlin(2019)·Liu(2019)의 Transformer 사전훈련, Park et al.(2021)의 한국어 NLI 벤치마크, Yin et al.(2019)의 zero-shot entailment 분류, Mohammad et al.(2016)의 stance detection, Ding et al.(2014, 2015)의 이벤트 기반 주가예측, Adams-MacKay(2007)의 Bayesian online changepoint detection, Bollen et al.(2011)의 텍스트→시장 인과 검정, Grootendorst(2022)의 c-TF-IDF 토픽 추출 기법을 결합·재구성하되, **단일 스칼라 비관·감성 점수를 6차원 정책 전달경로 증거 벡터로 확장하고, 검증된 시장 반응 방향을 앵커로 둔 설명모델 프레임에서 정책×채널 원자가설 NLI로 재구성한 PTEI(Policy Transmission Evidence Index)** 를 제안한다.

**허용 표현:** "독자적 분석 프레임워크", "정책-가격 전달경로 측정", "방향 고정 설명 모델", "기존 방법의 결합·재구성"
**금지 표현:** "세계 최초", "완전히 새로운 알고리즘", "검증된 방법론", "주가 예측 모델", "예측 정확도"

### 0.3 설계 원칙
1. 주가 반응 방향 `D_i`는 NLP가 추정하지 않는다. 서강대 CAR 산출물의 sign이 앵커.
2. 뉴스 점수 PTEI는 음수가 나오지 않는다 (§2.5).
3. 시간창은 사전에 모두 정의·동결한다 (§3). 메인은 단일 창.
4. 다중비교 보정은 Benjamini-Hochberg FDR로 통일. 메인 검정은 단일 사양이라 보정 불필요.
5. 정책별 최적창·임의 반응일 선택은 메인 결과에 사용하지 않는다. robustness로만.
6. **Sanity check는 파이프라인 디버깅 용도이며 성능 평가로 보고하지 않는다. 규칙 수정은 특정 케이스가 아니라 일반화 가능한 언어·정책 원칙에 한해 허용한다. 최종 성능 평가는 별도 holdout 라벨셋(test 300건)에서 수행한다.** (§12.12 참조)
7. **정책 카드 `policy_cards_v0.x`는 가설 템플릿 개발용이며 분석 결과 산출에는 사용하지 않는다. 분석에는 `review_status=ok`로 동결된 `policy_cards_v1.0`만 사용한다.** (§11.4 참조)
8. **정책별 가설은 case-by-case로 작성하지 않고, `type_subtype`·`D_i`·`industry`·`target_assets`·`active_channels`를 입력으로 하는 `build_hypothesis(card)` 함수로 자동 생성한다.** case별 수동 가설 작성·정책별 예외 템플릿 추가는 금지 (§12.14 참조).

---

## 1. 기준 데이터

본 방법론의 모든 처리는 다음 3개 파일을 단일 진실 원천(SSOT)으로 한다.

| 파일 | 역할 | 확인된 사실 |
|---|---|---|
| [50_전체기사_99539건.csv](50_전체기사_99539건.csv) | 본 분석 코퍼스 | 99,539건 / 102개 언론사 / rel_day [-30, +29] / 14컬럼 / 본문 200자cap |
| [50_전체기사_샘플3000건.xlsx](50_전체기사_샘플3000건.xlsx) | 라벨링 시드 | 2,913행 / 정책별 최대 60건 / 50정책 전부 포함 |
| [50_정책마스터.xlsx](50_정책마스터.xlsx) | 정책 메타·키워드·자연실험·언론사 매트릭스 | 14개 시트, 정책별 1·2차 키워드·NOT 제외어·ETF |

**확정 사양**
- 본문 200자cap. 풀텍스트 없음 — PPT-NLP의 단문 NLI 설계는 이 제약을 강점으로 전환 (§5).
- Tier-A 윈도우 D±30. Tier-B(D±90) 추후 확장.
- 정책별 기사 수 26 ~ 5,000건(상한 cap). N<300 정책은 §7.5 sparse 모드.

**고정 원칙**
1. 위 3개 파일 외 데이터 금지. 추가 시 §13 절차로 정책마스터 갱신.
2. 검증된 50건 주가 반응 방향(메모리: FDR 32/32, Naver 17/17 PASS) = 본 방법론의 앵커.

---

## 1.4 정책 카드 (Policy Card) — type_subtype 8개 동결 (v2.7 신설)

본 방법론에서 50건 정책 각각에 부여되는 메타 구조. case별 가설 작성을 막고 §12.14 build_hypothesis 자동 생성을 위한 입력.

### 1.4.1 type_subtype 8개 정의 (동결)

| subtype | 기본 D_i | 의미 | 예시 |
|---|---|---|---|
| `A_tax_support` | + | 세제 지원·자금 여력 개선 | K칩스법, 거래세 인하, 금투세 폐지 |
| `B_restriction_negative` | − | 규제 강화 — 산업 매출·심리 위축 | 부동산 규제, 의대증원, 의약분업 |
| `B_market_stabilizer` | + | 규제이지만 시장 안정·유동성 회복 목적 | 공매도 금지 4건 |
| `C_industrial_support` | + | 산업 육성·인프라·시장 신뢰 강화 | K-반도체 벨트, 보금자리, 바이오 육성 |
| `D_crisis_relief` | + | 위기 대응·정부 유동성 공급 | 100조 패키지, 소부장 대책 |
| `D_crisis_shock` | − | 위기 표면화·신뢰 훼손 | 황우석 폭로, ESS 화재, 5개 은행 퇴출 |
| `E_foreign_benefit` | + | 외국 정책 한국 수혜 | 미국 IRA |
| `E_foreign_shock` | − | 외국 정책 한국 충격 | 일본 수출규제, 미국 반도체 수출통제, IRA 축소, EU CBAM |

### 1.4.2 정책 카드 필드 사양 (v2.8 — D_i 3필드 구조)

`14_PolicyCard` 시트 (정책마스터 통합) 및 `policy_cards_v1.0.csv` 컬럼:

```
pn               : 정책 연번 (1-50)
industry         : 산업 (5개)
alias            : 정책 약칭
date             : 발표일 (YYYY-MM-DD)
master_type      : 정책마스터 ABCDE
type_subtype     : §1.4.1 8개 중 1개

D_i_initial      : ±1, v1.0 동결값 (라벨링·가설 생성용)
D_i_source       : D_i 초안 근거 ('policy_card_v1.0 (domain assessment)' 등)
D_i_final_car    : 서강대 최종 CAR 확정 후 채움 (현재 공란)

k_i              : 표준 D+5
C1~C6            : 활성 채널 ON/OFF
channel_signs    : 채널별 부호 (정책 강화/약화 방향)
rationale        : 1줄 근거
review_note      : 검토 근거
status           : v0.x_dev / v1.0_frozen
frozen_at        : 동결 일자
review_status    : ok / needs_review / conflict
```

**D_i 3필드 분리 원칙:** D_i는 본래 검증된 CAR sign이지만, 라벨링·가설 생성은 CAR 산출 전에 시작해야 하므로 도메인 판단 초안(D_i_initial)을 사용. 서강대 CAR 산출 후 D_i_final_car와 대조해 불일치 정책만 가설 재생성·NLI 재추론으로 보완.

### 1.4.3 버전 분리 원칙 (§0.3 #7 강제)

| 버전 | 용도 | 분석 사용 |
|---|---|---|
| `v0.x` | 가설 템플릿 개발·디버깅 | **금지** |
| `v1.0` | review_status=ok 동결본 | **유일한 분석 사용본** |

`conflict` 케이스는 별도 결정 트랙(서울경제 + 서강대)에서 정체 확정 후 처리.

### 1.4.4 v1.0 / v1.1 동결 (2026-05-16)

#### v1.0 (정정 6건 해소 동결)
- [policy_cards_v1.0.csv](policy_cards_v1.0.csv)
- 정책마스터 시트 `14_PolicyCard` 통합 (백업: `backups/50_정책마스터_*.bak`)
- [policy_hypotheses_v1.0.csv](policy_hypotheses_v1.0.csv)

**v0.2 → v1.0 정정 6건:**
- pn11 5개 은행 퇴출 → D_crisis_shock, D_i=−
- pn26 4·1 대책 → A_tax_support, D_i=+
- pn29 9·13 대책 → B_restriction_negative, D_i=− (2018 문 정부 9·13 확정)
- pn38 백신-CMO → D_crisis_relief, D_i=+
- pn39 의대증원 → B_restriction_negative, D_i=−
- pn40 CDMO 특별법 → A_tax_support, D_i=+

#### v1.1 (D-4·D-5 보정, v2.9)
- [policy_cards_v1.1.csv](policy_cards_v1.1.csv)
- [policy_hypotheses_v1.1.csv](policy_hypotheses_v1.1.csv)
- [channel_keywords_v0.2.json](channel_keywords_v0.2.json)
- 정책마스터 `14_PolicyCard` 시트 v1.1로 갱신

**D-5 subtype 활성 채널 재매핑** (정책 카드 channel ON/OFF 갱신):
```
A_tax_support              → [C1]              (C5 제외)
B_restriction_negative     → [C2, C4]
B_market_stabilizer        → [C5, C6]
C_industrial_support       → [C3, C6]
D_crisis_relief            → [C5, C4, C6]      (C4는 "불확실성 완화" 방향)
D_crisis_shock             → [C4, C6]          (C4는 "불확실성 확대" 방향)
E_foreign_benefit          → [C2, C3]
E_foreign_shock            → [C3, C4]          (C2 제외)
```

**D-4 channel_keywords v0.2** (C4 채널 일반 행정어 제거):
- 제거 from C4.neutral_anchors: 법, 법안, 제도, 정책, 행정, 심사, 승인, 인허가, 발표, 추진, 대책
- 보강 to C4.supports_higher_regulation: 위축, 갈등
- 보강 to C4.supports_lower_regulation: 안정, 신뢰회복

**D_crisis_relief vs D_crisis_shock 가설 분리:**
- relief C4: "시장 불안 완화·신뢰 회복" (호재 방향)
- shock C4: "패닉 확산·불확실성과 리스크 확대" (악재 방향)

**v1.1 sanity 결과 (디버깅 로그, 성능 보고 아님):**
- K칩스법 C4 활성률 0.83 → 0.50 (D-4 효과 명확, 33%p ↓)
- 일본 수출규제 C4 활성률 1.00 → 0.83
- K칩스법 C1 100% 유지 (A_tax_support → [C1] 매핑 정합)
- 공매도 금지 C5·C6 활성률(1.00, 0.67) 유지 (표면감성 우회 견고)
- 공매도 금지 C4 false active(1.00)는 본문 "금지" 단어 직접 매칭 — NLI stance가 분리 담당(§12.15 책임 분리)

### 1.4.5 Dev/Test stratified split (v2.8 신설)

`50_전체기사_샘플3000건.xlsx` 2,913건을 정책×시간버킷(pre/d0/post) stratified로 분리:

| split | 건수 | 용도 | 잠금 |
|---|---|---|---|
| Sanity | 10 | 파이프라인 디버깅 | sanity_check_cases.json |
| **Dev** | **100** | 라벨링 가이드·임계·가설 조정 | splits/dev_split.csv |
| **Test** | **300** | **최종 평가 1회 — 분석 종료까지 잠금** | splits/test_split.csv + LOCKED.md |
| Train | 2,513 | 라벨링 풀 (잔여) | splits/train_split.csv |

- 정책별 균등: Dev 평균 2.0건/정책, Test 평균 6.0건/정책, 0건 정책 없음
- 시간 버킷 분산: pre (rel_day<0) / d0 (rel_day∈[0,1]) / post (rel_day>1) round-robin 추출
- random seed 42, 재현 스크립트: [build_v1_pipeline.py](build_v1_pipeline.py)
- Test set 잠금 강제: [splits/test_split.LOCKED.md](splits/test_split.LOCKED.md)

---

## 1.5 프로젝트 고유 변형 — "우리만의 것" (대외 인용용)

본 방법론이 선행연구에서 직접 가져오지 않은, **본 프로젝트가 처음 도입한 7가지 변형**:

| # | 변형 | 선행연구 등가물 | 본 방법론의 변형 |
|---|---|---|---|
| 1 | **채널 벡터 6차원** | Tetlock 2007: 단일 pessimism factor. L-McDonald 2011: 단일 net tone | 정책×채널 6차원 시그너처 (정책별 dominant channel 식별 가능) |
| 2 | **정책 전달경로 카테고리** | L-McDonald 2011: 단어 감성 6분류 (Negative·Positive·Uncertainty·Litigious·Modal) | 정책 효과 6경로 (비용·수요·공급·규제·유동성·심리) — 감성이 아닌 전달 메커니즘 |
| 3 | **D_i 앵커 + 정렬 PTEI** | Tetlock 2007·Bollen 2011: 텍스트·주가 둘 다 변수 (예측 검정) | 검증된 주가 방향을 사후 앵커로 두고 NLP를 종속 설명변수로 위치 |
| 4 | **채널별 원자 가설 H_{i,c}** | Yin et al. 2019: 단일 zero-shot 가설 entailment | 정책 50 × 채널 평균 3 = ~150개 원자가설 자동 생성 후 다중 NLI |
| 5 | **diffusion (언론사 다양성)** | (직접 등가물 미발견 — 본 프로젝트 고유 운영지표) | 일별 `1 - max_provider_share` — 보도자료 복붙·단일사 독점 흡수 |
| 6 | **4유형 분류 (peak_day 기반)** | (직접 등가물 미발견 — 본 프로젝트 고유 운영지표) | 즉시·선반영·지연·설명보강 — 50건 전수 배정 강제 |
| 7 | **Support/Contradiction paired 검정** | Tetlock 2007: net score만 검정 | 두 지표를 분리 측정해 paired Wilcoxon으로 차이 검정 |

이 7가지가 본 프로젝트가 단순 "감성분석"이나 "NLI 분류"와 구별되는 지점이다. 신청서·5부작 기사에서 PTEI를 설명할 때 이 7가지 중 1~2개를 핵심으로 거론한다.

---

## 2. 핵심 변수 정의

### 2.1 정책 검증 주가 방향 `D_i` (앵커)

```text
k_i  = 정책 i의 사전 고정 반응창 (표준 D+5)
D_i  = sign(CAR_{i, [0, k_i]})  ∈ {+1, -1}
```

`D_i`는 서강대 이벤트 스터디 산출물에서 직접 가져온다. NLP 추정 금지.

### 2.2 6개 표준 전달경로

| 코드 | 채널 | 의미 | 키워드 시드 |
|---|---|---|---|
| C1 | 비용/세부담 | 세액공제·보조금·원가 | 세액공제, 감면, 세부담, 원가, 비용 |
| C2 | 수요/매출 | 매출·거래·분양·처방 | 수요, 매출, 분양, 거래, 주문 |
| C3 | 공급/생산능력 | 설비·공급망·국산화 | 설비, 공급, 생산, 증설, 국산화 |
| C4 | 규제/불확실성 | 규제·리스크 | 규제, 금지, 강화, 불확실, 리스크 |
| C5 | 자금조달/유동성 | 자금·유동성·거래안정 | 유동성, 자금, 조달, 거래안정, 공매도 |
| C6 | 시장신뢰/투자심리 | 신뢰·거버넌스·평가 | 신뢰, 심리, 거버넌스, 평가, 등급 |

### 2.3 정책 카드 `Card_i`

```text
Card_i = {
  D_i           : sign(CAR_{i, k_i}),
  k_i           : 사전 고정 반응창 (표준 D+5),
  C_set_i       : 활성 채널 집합 ⊂ {C1..C6},
  sign_per_c_i  : c ∈ C_set_i 각각의 방향부호
}
```

정책마스터 신규 시트 `14_PolicyCard`에 저장 (50행). 라벨링 착수 전 v1 동결.

### 2.4 채널별 원자 가설 `H_{i,c}` (지지/반박/중립)

각 정책·채널 쌍에 대해 NLI 가설 자동 생성:

```text
H_{i,c} := "이 기사는 정책 {정책명}이(가) {산업} 주가에
            {채널 c의 의미}를 통해 {D_i · sign_per_c} 방향으로 작용한다고 설명한다."
```

NLI 모델 출력: `(p_support, p_neutral, p_contradict)`, 합=1.

### 2.5 기사별 PTEI — 방향 고정 설명 점수 (음수 봉쇄)

PTEI는 0 이상 한 방향만 측정. **v2.5 핵심 변경 + v2.7.1 책임 분리 명문화:**
- `channel_match`를 multiplier → **indicator gate `I(channel_match ≥ τ_c)`**로 강등 (NLI 가설 H_{i,c}가 이미 채널을 명시하므로 이중 가중 방지)
- `novelty`는 기사 PTEI에만 두고 일별 NP에서 제거 (일별 mean으로 자동 반영, 이중 반영 방지)
- **NLI는 채널 분류기가 아니다.** NLI 출력은 각 채널 가설에 대한 stance 확률(`p_support`, `p_contradict`)로만 사용하며, 채널 활성 여부는 channel_match gate가 결정한다. (§12.15 책임 분리 원칙, C-1 sanity 결과 반영)

#### 채널 게이트 (v2.7.1 — indicator로 명문화)
```text
gate(a, c) = I(channel_match_{a,c} ≥ τ_c)   ∈ {0, 1}
τ_c        : 채널별 임계 (dev에서 calibrate, 기본 0.25)
I(·)       : indicator 함수 — 점수가 애매하게 전부 조금씩 들어가는 문제 방지
```

채널 c와 어휘적·의미적으로 무관한 기사는 NLI 가설을 적용해도 노이즈만 추가하므로 사전 제거. **NLI는 stance만, gate가 채널을** 담당.

#### 기사별 점수
```text
PTEI_{a,c}          = relevance_a × I(channel_match_{a,c} ≥ τ_c) × p_support_{a,c}    × specificity_a × novelty_a
Contradiction_{a,c} = relevance_a × I(channel_match_{a,c} ≥ τ_c) × p_contradict_{a,c} × specificity_a × novelty_a

PTEI_a          = Σ_{c ∈ C_set_i} PTEI_{a,c}
Contradiction_a = Σ_{c ∈ C_set_i} Contradiction_{a,c}
```

해석: PTEI_a는 "이 기사가 검증된 주가 방향(`D_i`)을 채널 c를 통해 얼마나 강하게 지지하는가"의 강도. 음수 안 나옴.

**Robustness:** gate 대신 weak prior 변형 `mult = α + (1-α) × channel_match_{a,c}` (α=0.7)로 비교 보고. 두 변형의 H1·H2 결과 차이를 §7.6에 부록.

| 항목 | 산출 방식 | 범위 |
|---|---|---|
| `relevance_a` | 정책마스터 1·2차 키워드 매칭 | [0, 1] |
| `gate(a,c)` | channel_match ≥ τ_match 이진 게이트 | {0, 1} |
| `p_support_{a,c}` | NLI(전제=[제목+본문], 가설=H_{i,c}) | [0, 1] |
| `p_contradict_{a,c}` | 동일 NLI contradict 확률 | [0, 1] |
| `channel_match_{a,c}` | 채널 c 시드와의 cos-sim (**gate 산출 전용**, 점수 곱셈 미사용) | [0, 1] |
| `specificity_a` | NER + 정규식 (종목·수치·기관·금액) | [0.5, 1.5] |
| `novelty_a` | 1 − max cos(임베딩 a, D-1일 기사들). **기사 PTEI에만 사용 (일별 NP는 별도 곱하지 않음)** | [0, 1] |

### 2.6 일별 뉴스 압력 `NP` (Support와 Contradiction 분리)

**v2.5 변경:** `novelty`는 §2.5의 기사 PTEI에 이미 포함되어 mean(PTEI_{a,c})을 통해 일별로 자동 평균됨. 일별 NP에서 별도 곱하지 않음 (이중 반영 해소).

```text
NP_support_{i,t,c}     = mean(PTEI_{a,c})          × log(1+N_{i,t}) × Diffusion_{i,t}
NP_contradict_{i,t,c}  = mean(Contradiction_{a,c}) × log(1+N_{i,t}) × Diffusion_{i,t}

where Diffusion_{i,t} = 1 - max_provider_share_{i,t}
      N_{i,t}         = article count for policy i on rel_day t
```

Support는 메인 분석에, Contradiction은 §7.1·§7.3 비교 검정에 사용.

### 2.7 정렬된 주가 측도

```text
CAR_aligned_{i,t}  = D_i × CAR_{i,t}
```

`D_i`가 검증된 앵커이므로 `CAR_aligned_{i, k_i}`는 정의상 `≥ 0`.

---

## 3. 시간창 설계 — 메인 단일 고정창

### 3.1 메인 분석창 (모든 정책 동일, 사후 변경 금지)

```text
뉴스 메인창:  D-7 ~ D+5    (NewsFlow_main)
주가 메인창:  D0  ~ D+5    (CAR_main)
```

근거:
1. D-7~D-1: 정책 전 선반영
2. D0~D+5: 정책 직후 해석·반응
3. D+5 이후: 다른 이벤트 혼입 위험 ↑

메인 통계는 이 단일 창 한 쌍만 사용. 정책별 임의 변경 금지.

### 3.2 메인 지표 계산

```text
NewsFlow_main_i      = Σ NP_support_{i,t}        t ∈ [-7, +5]
ContradictionFlow_i  = Σ NP_contradict_{i,t}     t ∈ [-7, +5]
CAR_main_i           = CAR_aligned_{i, [0, +5]}  (= D_i × CAR_{i, [0, +5]})
```

설계상 `NewsFlow_main_i ≥ 0`, `CAR_main_i ≥ 0` (검증된 50건). **방향 일치는 정의로 보장.**

### 3.3 보조 구간 (서사·robustness)

| 구간 | 범위 | 용도 |
|---|---|---|
| W0 | D-30 ~ D-8 | 배경 예열 (서사용) |
| W1 | D-7 ~ D-1 | 선반영만 분해 |
| W2 | D0 ~ D+1 | 즉시반응만 분해 |
| W3 | D+2 ~ D+5 | 단기해석만 분해 |
| W4 | D+6 ~ D+10 | 지연반응 (4유형 분류용) |
| W5 | D+11 ~ D+29 | 사후확산 (서사용) |

메인 결과 보고에는 W1~W3 합산(=메인창)만 사용. W0·W4·W5는 robustness 부록.

---

## 4. 50건 4유형 분류 (지지 근거 형성 시점)

방향 일치는 정의로 보장되므로, 분류 기준은 "지지 근거가 언제 형성됐는가"이다.

```text
peak_day_i = argmax_t NP_support_{i,t}    t ∈ [-7, +10]
```

| 유형 | 판정 규칙 | 의미 |
|---|---|---|
| **A. 즉시반응형** | peak_day_i ∈ [0, +5] AND NewsFlow_main_i ≥ 코호트 중위 | 정책 직후 지지 근거 집중·시장도 동시 반응 |
| **B. 선반영형** | peak_day_i ∈ [-7, -1] AND `|CAR_{i,0}|` < 코호트 중위 | 시장이 정책 전 이미 가격에 반영 |
| **C. 지연반응형** | peak_day_i ∈ [+6, +10] AND `CAR_{i,[+6,+10]}` 부호가 D_i | 정책 해석이 늦게 가격에 반영 |
| **D. 설명보강형** | NewsFlow_main_i < 코호트 25%ile | 정량은 약하지만 PTEI 상위 5건이 D_i 채널에 집중 |

우선순위 A → B → C → D. 50건 전수 배정.

기사 서사 핵심 문장:

> 50개 정책 모두에서 검증된 주가 반응 방향을 설명하는 정책 전달경로가 뉴스 안에서 확인됐다. 다만 그 근거가 형성된 시점은 즉시·선반영·지연·설명보강 4유형으로 나뉜다.

---

## 5. 라벨링 설계

샘플 2,913건은 다음 2개 항목만 사람이 라벨링한다.

| 라벨 | 값 | 정의 | 자동/수동 |
|---|---|---|---|
| `relevance` | 2 / 1 / 0 | 정책 직접 / 배경 / 무관 | **수동** |
| `stance_label` | support / contradict / neutral | `D_i` 방향 지지 / 반박 / 중립 | **수동** |
| `channel` | C1..C6 multi-label | 채널 키워드·임베딩 매칭 | 자동 |
| `specificity` | [0.5, 1.5] | NER+정규식 | 자동 |
| `novelty` | [0, 1] | D-1 기사 cos-distance | 자동 |

라벨링 규칙:
1. 기사 톤이 아니라 **정책 대상 종목/산업의 검증된 주가 방향 `D_i`에 우호적인지** 판단.
2. `D_i`와 같은 해석 = support, 반대 = contradict, 무방향 = neutral.
3. `relevance=0`이면 stance 면제.
4. 2명 독립 라벨링, Cohen's κ ≥ 0.7.

**모델 학습 타깃은 3분류 (support/neutral/contradict).** 추론 시점에 `p_support`와 `p_contradict`를 따로 뽑아 §2.5·2.6에 투입.

---

## 6. 모델 구조

### 6.1 베이스라인 비교 (3-tier)

| Tier | 방법 | 역할 |
|---|---|---|
| T0 | 키워드/사전 규칙 (KNU + 채널 시드 + L-McDonald 번역) | 하한선·오류진단 |
| T1 | KoBERT/KoFinBERT 파인튜닝 (3분류) | 회계학연구 2022 재현 |
| T2 | **KLUE-RoBERTa-large + KLUE-NLI 체크포인트 → stance 3분류 fine-tuning** | 최종 후보 |

dev macro-F1 최고 채택. 1%p 이내면 T2.

### 6.2 모델 입력

```text
Premise:    [제목] [SEP] [본문 200자] [SEP] [정책명] [SEP] [산업]
Hypothesis: H_{i,c}  (§2.4)
Output:     softmax(support, neutral, contradict)
```

max_len = 256.

### 6.3 추론 최적화 — relevance gate

```text
if relevance_a < 0.3:   # 키워드 매칭 점수
    skip NLI; PTEI_a = 0
else:
    run NLI on (a) × C_set_i
```

99,539건 × 평균 3채널 = ~300K → relevance gate로 ~30% 감축, GPU 1대 4~5시간 → 3~4시간.

### 6.4 왜 NLI 구조인가

| 항목 | 단일 stance 분류기 | 채널별 NLI |
|---|---|---|
| 출력 | s ∈ [-1, +1] 1개 | 채널 벡터 6차원 |
| 5부작 활용 | "톤이 호의적" | **"시장은 X 경로에 반응했다"** |
| 200자 적합성 | 의미 추론 한계 | 짧은 stance 판단에 최적 |
| 공매도 같은 경계 사례 | 단어 표면에 흔들림 | 채널 정의가 모호성 흡수 |

---

## 7. 검정 방법 — 강도·타이밍·위약 (방향 일치 검정 제거)

본 v2.3에서는 부호 일치 검정을 메인에서 제거한다. 방향 일치는 §2.5·2.6 설계로 보장되므로, 검정은 **(a) 지지 근거의 강도, (b) 형성 시점, (c) 우연 대비 우월성**을 본다.

### 7.1 H1 — 지지 근거 > 반박 근거 (paired)

```text
H1: NewsFlow_main_i > ContradictionFlow_main_i  for i=1..50
```

검정:
- Wilcoxon signed-rank (n=50, paired)
- Cohen's d (효과 크기)
- bootstrap 95% CI

**해석:** 검증된 주가 방향을 지지하는 뉴스 근거가 반박 근거보다 강하다.

### 7.2 H2 — PTEI 강도 ↔ CAR 크기 회귀

```text
CAR_main_i = β₀ + β₁ · NewsFlow_main_i
           + β₂ · log(1 + ArticleCount_i)
           + β₃ · (NewsFlow_main_i - ContradictionFlow_main_i)
           + Industry_FE + PolicyType_FE + ε_i
```

기대: `β₁ > 0`, `β₃ > 0`. heteroskedasticity-robust SE.

**해석:** 뉴스 근거가 강한 정책일수록 CAR 절댓값도 크다 (지지 강도 ↔ 시장 반응 크기).

### 7.3 H3 — 위약(placebo) 비교 (v2.5 재정의)

**v2.5 변경:** v2.4의 "D±30 내부 random 14-day 100회"는 (a) 60일 내 14일창 100개의 독립성 약함, (b) event window와 overlap 가능. 2-track 설계로 교체.

#### Track 1 — 먼 구간 within-policy placebo
정책 i의 D-30~-16 (15일) 및 D+16~+29 (14일) 먼 구간 NP_support 평균 vs 메인창 평균:

```text
NP_placebo_far_i = mean(NP_support_{i,t}) for t ∈ [-30,-16] ∪ [+16,+29]   (29일)
NP_event_i       = NewsFlow_main_i / 13                                     (메인창 일평균)

H3a: NP_event_i > NP_placebo_far_i × c     (c=1.5 기본; calibration robustness c∈{1.2,1.5,2.0})
```

50건 paired Wilcoxon. 메인창과 먼 구간이 시간적으로 분리되어 overlap 위험 없음.

#### Track 2 — Cross-policy permutation
정책 i의 NP 시계열을 다른 정책 j(j≠i)의 0일축에 매핑하여 가상 NewsFlow_main 생성:

```text
NewsFlow_perm_{i ← j} = Σ_{t∈[-7,+5]} NP_support_{i,t}  (j의 정책일을 0으로 재정렬)
NewsFlow_event_i      = NewsFlow_main_i

empirical_p_i = (1/49) × Σ_{j≠i} 1[ NewsFlow_perm_{i←j} ≥ NewsFlow_event_i ]
```

- 50건 × 49 = 2,450 permutation
- 50건 BH-FDR(q<0.10) 보정
- 50건 binomial: empirical_p<0.10 정책 비율 > 0.5

#### 통합 검정
- Track 1 통과 + Track 2 통과 → 강건 (메인 결과로 채택)
- 한쪽만 통과 → 부분적 (보조 결과)
- 둘 다 실패 → 정책 i를 4유형 D(설명보강형)으로 강제 배정

**해석:** 두 트랙 모두 정책일 주변 지지 근거가 우연(먼 구간 또는 다른 정책의 동일 시간 패턴) 대비 강하다는 것을 확인. 단일 random window 검정보다 독립성·해석 모두 단단.

### 7.4 H4 — 타이밍 정렬

```text
peak_day_i  = argmax_t NP_support_{i,t}
react_day_i = argmax_t |CAR_aligned_{i,t}|     t ∈ [0, +10]
gap_i       = react_day_i - peak_day_i
```

검정:
- `|gap_i|` 분포 (50건 히스토그램)
- Wilcoxon: `|gap_i|` 중위값 < 3일?
- pre-AI vs post-AI 비교 (5편 기사용)

**해석:** 뉴스 근거 peak와 주가 반응 day가 근접 (시간 정렬).

### 7.5 H5 — Cross-correlation Lag (Layer 3 / 4편)

```text
lag ∈ [-7, +10]
corr_i(lag) = corr(NP_support_{i,t}, R_aligned_{i, t+lag})
best_lag_i  = argmax_lag corr_i(lag)
```

50건 best_lag 분포 → 정책 유형별 선행성 평균. Granger는 N≥1000 정책에 한해.

**해석:** 뉴스가 주가를 선행하는가, 주가가 뉴스를 선행하는가의 정량 근거.

### 7.6 보조 — Robustness (탐색적 결과, FDR 보정)

후보 구간 풀(N1~N6 × R1~R5 = 30조합)에서 정책별 best-window 탐색. **메인 결론에 사용하지 않음**. 5부작 보조 서사·정책별 1-pager에만.

```text
WindowScore_{i,k,m} = (NewsFlow_in_N_k - Contradiction_in_N_k)
                    × CAR_aligned_in_R_m
                    × log(1+article_count_in_N_k)
                    × penalty(window_length)
penalty(L) = 1/sqrt(L)
```

50 × 30 = 1,500 검정 → BH-FDR(q<0.10).

### 7.7 Sparse 모드

| N | 가용 분석 |
|---:|---|
| `N ≥ 1000` | H1·H2·H3·H4·H5 모두 |
| `500 ≤ N < 1000` | H1·H2·H3·H4 (Granger 제외) |
| `300 ≤ N < 500` | H1·H2·H3만 |
| `N < 300` | H1·H2 풀링에 포함, D유형 강제 배정 (시밀러가이드 26·CDMO 40 등) |

### 7.8 채널 시그너처 — 5부작 4편 핵심

```text
dominant_c_i = argmax_c NP_support_{i, [-7,+5], c}
```

50 × 6 매트릭스에서 정책 유형별 채널 분포:
- A 세제 → C1 우세
- B 규제 → C4·C2 우세
- C 산업육성 → C3·C6 우세
- D 위기 → C5·C4 우세
- E 외국 → C3·C4 우세

→ "시장은 비용 경로에 반응했다 vs 공급망 경로에 반응했다" 식 서사.

---

## 8. 검정 결과 보고 포맷

### 8.1 메인 통계 (사전 등록, 단일 사양)

```text
H1 Support > Contradiction:  Wilcoxon W=..., p=..., d=...
H2 회귀 β₁:                  β=..., t=..., p=..., R²=...
H3 위약 우월:                event > 95%ile 정책 X/50건, FDR p=...
H4 타이밍 gap:               median |gap|=..., Wilcoxon p=...
H5 best_lag 분포:            median=..., IQR=...
```

### 8.2 50건 유형 분류표

| # | 정책 | D_i | NewsFlow_main | CAR_main | peak_day | react_day | 유형 | dominant_c |
|---|---|---|---|---|---|---|---|---|
| 1 | 반도체 빅딜 | -1 | 0.42 | 0.038 | -2 | +3 | B 선반영 | C5 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 8.3 학술 정직성 분리

```text
[메인 결과]  : 사전 등록 단일 창 (D-7~D+5 / D0~D+5)
              방향 일치는 설계상 보장. 검정은 강도·타이밍·위약.

[탐색 결과]  : 30 후보창 자동탐색, FDR<0.10 보고
              유형 분류·1-pager에만 사용. 메인 결론 불용.
```

---

## 9. 전처리 원칙

1. **중복 제거** — `news_id` 중복 1건 / 동일 정책 내 제목 MinHash > 0.85 → 본문 최장 1건 유지
2. **무관 기사 필터** — 정책마스터 `NOT 제외어` + 1·2차 키워드 미포함 → `relevance=0`
3. **언론사 가중** — 일별 단일사 점유 > 30% 시 `w = 1/√n`
4. **기사량 보정** — `log(1+N)`

---

## 10. 산출물

### 10.1 데이터
- `data/processed/policy_cards.parquet` — 50건 정책 카드
- `data/processed/article_scores.parquet` — 99,539건 × 채널 6 PTEI + Contradiction
- `data/processed/daily_panel.parquet` — 정책 50 × 일자 60 × 채널 6 패널
- `data/processed/policy_type_classification.csv` — 4유형 분류

### 10.2 보고서
- `reports/PTEI_main.md` — H1~H5 메인 검정
- `reports/per_policy_cards/` — 정책별 1-pager (카드+채널+유형+근거기사 Top5)
- `reports/channel_signature.md` — 정책 유형별 dominant 채널 매트릭스
- `reports/robustness/` — 윈도우·임계·가중·30조합 sensitivity

### 10.3 5부작 기사 매핑

| 편 | 핵심 산출물 |
|---|---|
| 1편 충격 Top5 | NewsFlow_main 상위 5건 + 4유형 A·B |
| 2편 선반영 | 유형 B 정책 전수 + W1 NP_support 시계열 |
| 3편 논조 전환 | peak_day 시점 채널 시그너처 전환 |
| 4편 인과방향 | H5 best_lag 분포 (50건) |
| 5편 AI 시대 | pre/post-AI H4 gap 비교 |

---

## 11. 인용 논문 — PTEI 구성요소별 매핑 (papers/ 폴더 19편)

### 11.1 PTEI 구성요소 ↔ 논문 매핑

| 구성요소 | 직접 베이스 (논문) | 그 논문이 제시하는 것 | **본 방법론의 변형** |
|---|---|---|---|
| **6채널 카테고리** | [L-McDonald 2011](papers/LoughranMcDonald_2011_LiabilityDictionary_JF.pdf) + [Baker-Bloom-Davis 2016](papers/BakerBloomDavis_2016_EconomicPolicyUncertainty_QJE.pdf) | LM: 단어 감성 6분류 / BBD: 정책 불확실성 신문빈도 차원 | 감성/불확실성 → **정책 전달경로 6채널** 재범주화 |
| **채널 키워드 시드** | [L-McDonald 2011](papers/LoughranMcDonald_2011_LiabilityDictionary_JF.pdf) + [Grootendorst 2022](papers/Grootendorst_2022_BERTopic_arxiv2203.05794.pdf) + [Blei et al. 2003](papers/BleiNgJordan_2003_LDA_JMLR.pdf) | 수동 사전 + c-TF-IDF + LDA 토픽 | 한국어 정책 시드 수동 + BERTopic 자동 확장 |
| **relevance 필터** | [Baker-Bloom-Davis 2016](papers/BakerBloomDavis_2016_EconomicPolicyUncertainty_QJE.pdf) | "Economic + Policy + Uncertainty" AND 매칭법 | 정책마스터 1·2차 키워드 + AND/NOT |
| **stance NLI 구조** | [Mohammad et al. 2016 SemEval](papers/Mohammad_2016_SemEvalStanceDetection_ACL.pdf) + [Yin-Hay-Roth 2019](papers/YinHayRoth_2019_ZeroShotEntailment_arxiv1909.00161.pdf) | stance detection task + zero-shot entailment 분류 | 채널별 원자가설 NLI |
| **NLI 베이스 모델** | [Devlin et al. 2019 BERT](papers/Devlin_2019_BERT_NAACL_N19-1423.pdf) + [Liu et al. 2019 RoBERTa](papers/Liu_2019_RoBERTa_arxiv1907.11692.pdf) + [Park et al. 2021 KLUE](papers/Park_2021_KLUE_arxiv2105.09680.pdf) | Transformer 사전훈련 + 한국어 NLU 벤치마크·체크포인트 | KLUE-RoBERTa-large + KLUE-NLI 체크포인트 |
| **도메인 적응 3단계** | [Araci 2019 FinBERT](papers/Araci_2019_FinBERT_arxiv1908.10063.pdf) + [Yang 2020 FinBERT](papers/Yang_2020_FinBERT_arxiv2006.08097.pdf) + [Hiew 2019](papers/Hiew_2019_BERTFinancialSentimentIndex_arxiv1906.09024.pdf) | BERT → 금융 MLM → fine-tune (86%→97%) | KLUE-RoBERTa → 99K 경제기사 MLM → stance fine-tune |
| **한국어 금융 도메인** | [TWICE 2025](papers/TWICE_2025_KoreanFinancial_arxiv2502.07131.pdf) | 한국어 저자원 금융 임베딩 | channel_match cos-sim 베이스 임베딩 |
| **novelty (참신성)** | [Tetlock 2007](papers/Tetlock_2007_GivingContentInvestorSentiment_JF.pdf) + [Xu-Cohen 2018](papers/XuCohen_2018_StockMovementTweetsPrices_ACL_P18-1183.pdf) | AR(5) residual / tweet novelty | D-1 임베딩 cosine distance |
| **비관 측정 → 채널 벡터** | [Tetlock 2007](papers/Tetlock_2007_GivingContentInvestorSentiment_JF.pdf) | WSJ pessimism factor (단일 스칼라) | **단일 → 6차원 채널 벡터** (핵심 변형) |
| **이벤트 기반 주가 모델링** | [Ding 2014 EMNLP](papers/Ding_2014_StructuredEventsStockMovement_EMNLP_D14-1148.pdf) + [Ding 2015 IJCAI](papers/Ding_2015_DeepLearningEventDrivenStockPrediction_IJCAI.pdf) | 구조화 이벤트(O, A, P) tuple → 주가 예측 | 정책을 50개 구조화 이벤트로 모델링 |
| **텍스트→주가 인과 검정** | [Tetlock 2007](papers/Tetlock_2007_GivingContentInvestorSentiment_JF.pdf) + [Bollen et al. 2011](papers/BollenMaoZeng_2011_TwitterStockMarket_arxiv1010.3003.pdf) + [Xu-Cohen 2018](papers/XuCohen_2018_StockMovementTweetsPrices_ACL_P18-1183.pdf) | VAR / Granger / tweet→stock | PTEI 시계열 × CAR VAR(5), 정책별 best_lag |
| **변화점·peak_day 탐지** | [Adams-MacKay 2007](papers/AdamsMacKay_2007_BayesianOnlineChangepoint_arxiv0710.3742.pdf) | Bayesian Online Changepoint Detection | NP_support 시계열 변화점 → peak_day |
| **시간창 D-7~D+5** | [Tetlock 2007](papers/Tetlock_2007_GivingContentInvestorSentiment_JF.pdf) (5-day VAR) + [L-McDonald 2011](papers/LoughranMcDonald_2011_LiabilityDictionary_JF.pdf) ([0,+3]) + [Baker-Bloom-Davis 2016](papers/BakerBloomDavis_2016_EconomicPolicyUncertainty_QJE.pdf) | 학술 표준 단기 윈도우 | 50건 코퍼스 EDA로 [-7,+5]가 정점 20% 영역 검증 |
| **β₁ > 0 회귀** | [Tetlock 2007](papers/Tetlock_2007_GivingContentInvestorSentiment_JF.pdf) + [L-McDonald 2011](papers/LoughranMcDonald_2011_LiabilityDictionary_JF.pdf) | 미디어 비관→수익률 음의 회귀 / 10-K 부정어→발표창 수익률 | 동일 구조, 단 PTEI는 사후 정렬된 양의 신호 |
| **위약(placebo)·다중비교** | (이벤트 스터디 일반) + [Bollen 2011](papers/BollenMaoZeng_2011_TwitterStockMarket_arxiv1010.3003.pdf) | random window 비교 | **2-track: 먼 구간 within-policy + cross-policy permutation (50×49)** + BH-FDR |
| **미디어 빈도 가중 log(1+N)** | Hillert-Jacobs-Müller 2014 (RFS, paywall — papers/ 미수록, [SSRN 2023442](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2023442) 인용만) | "Media Makes Momentum" — 보도량이 momentum 강화 | log(1+N) 일별 가중치 |
| **한국어 도메인 베이스라인** | 현지원·이준일·조현권 (2022) 회계학연구 47(4) — KCI ([링크](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002873618), KCI는 직접 다운로드 미지원) | KoBERT 한국 기업뉴스 감성 85.7% | T1 베이스라인 재현 |

### 11.2 papers/ 폴더 파일 인덱스 (전체 19편)

핵심·기존 (v2.0): Tetlock 2007, L-McDonald 2011, Araci 2019 FinBERT, Yang 2020 FinBERT, Park 2021 KLUE, Grootendorst 2022 BERTopic, TWICE 2025

확장·v2.4 추가: Devlin 2019 BERT, Liu 2019 RoBERTa, Blei 2003 LDA, Baker-Bloom-Davis 2016 EPU, Adams-MacKay 2007 Changepoint, Ding 2014/2015 이벤트 주가, Xu-Cohen 2018 트윗 주가, Hiew 2019 BERT Financial Sentiment, Bollen 2011 Twitter Stock, Yin-Hay-Roth 2019 Zero-shot NLI, Mohammad 2016 SemEval Stance

---

## 12. 추가 강화 고려 사항 (v2.4 신설)

본 방법론을 더 단단하게 만들기 위한 보조 설계. 각 항목은 메인 방법론과 독립적으로 채택·연기 가능.

### 12.1 모델 해석성 (SHAP / attention)
**왜:** 5부작 기사에서 "왜 이 기사가 호재로 분류됐는가" 예시 제공이 핵심 서사. 단순 점수만으로는 독자 설득력 약함.
**방법:**
- SHAP values: 각 기사의 NLI 결정에서 토큰별 기여도
- Attention map: KLUE-RoBERTa의 [CLS] 토큰이 본문에서 주목한 단어
- 정책별 dominant 단어 Top10 추출
**근거:** Lundberg & Lee 2017 SHAP (별도 검색 가능)
**우선순위:** 높음 (기사 서사 직결)

### 12.2 본문 200자 → 풀텍스트 보강
**왜:** 현재 코퍼스는 200자cap. NLI의 미세한 stance 판단에 한계. 단 [reference_bigkinds_api.md](/Users/somv/.claude/projects/-Users-somv-Documents---------/memory/reference_bigkinds_api.md)의 서울경제 전용 키(254bec69)는 풀텍스트 제공.
**방법:**
- 서울경제 기사 5,094건만 풀텍스트 재수집 → 별도 컬럼 `content_full` 추가
- NLI 입력 시 풀텍스트 우선, 부재 시 200자
- 99K 전체는 200자 유지 (수집 비용)
**우선순위:** 중간 (서울경제 비중 5%로 한계, 단 신뢰성 향상)

### 12.3 시기별 어휘 drift 처리
**왜:** 1998 IMF 시절 "구조조정" 어휘와 2025 "K칩스법" 어휘는 의미장이 다름. 단일 사전이 28년 시계열에 부적합.
**방법:**
- 5년 단위 슬라이딩 윈도우로 도메인별 BERTopic 추가 학습
- channel_keywords.json을 시기별 버전 관리 (`_1998`, `_2010`, `_2025`)
- 또는 시기를 입력 메타데이터로 NLI에 주입 ([제목][SEP][본문][SEP][정책연도])
**근거:** Ding 2014/2015 시계열 사전 + Tetlock의 시기 통제
**우선순위:** 중간 (1998~2005 9건 영향)

### 12.4 외생 충격 통제 (시장 전체 충격 분리)
**왜:** 리먼(2008.9.15), 코로나(2020.3) 직후 정책은 정책 효과와 시장 패닉이 혼재. CAR_aligned 자체가 오염됨.
**방법:**
- KOSPI 일별 −2%+ 또는 +2%+ 일자를 외생 충격일로 표기
- 정책 i의 D±5 안에 외생 충격일이 있으면 robustness 별도 보고
- 시장 모델 abnormal return (Fama-MacBeth) 대신 사용 가능 (서강대 결정)
**우선순위:** 높음 (리먼·코로나 사례 다수)

### 12.5 그레이존 정책 처리 (공매도금지 케이스)
**왜:** B(규제)면서 D_i=+1인 정책(공매도 금지 4건). 채널 부호(`sign_per_c`)가 일반 B 정책과 반대.
**방법:**
- 정책 카드 `14_PolicyCard` 시트에 `is_inverted_B` 플래그 추가
- NLI 가설 생성 시 채널 부호 명시적 반영
- 5부작 별도 코너 "위기 시 규제는 호재" 사례 분석
**우선순위:** 중간 (해당 정책 4건, 명시 처리 필요)

### 12.6 Sparse 정책 베이지안 풀링
**왜:** N<300 정책 4건 (시밀러 가이드 26·CDMO 40·첨단전략 27·반도체촉진 77)은 자체 검정 불가. 그렇다고 D유형 강제 배정만 하면 정보 손실.
**방법:**
- 산업·정책유형 hierarchy 베이지안 모델로 풀링
- `PTEI_i ~ Normal(μ_industry_type, σ²)` 형태로 sparse 정책의 점수 추정에 prior 활용
- Stan 또는 PyMC로 구현
**근거:** Gelman et al. 베이지안 계층 모델
**우선순위:** 낮음 (4건만 영향)

### 12.7 거래일 변환 (rel_day 캘린더 → 거래일)
**왜:** 현재 rel_day는 캘린더일. 금요일 정책 발표 시 D+1은 토요일(시장 닫힘) → D+3 월요일에야 시장 반응. 단순 비교 시 오해 가능.
**방법:**
- KRX 거래일 캘린더로 캘린더일 ↔ 거래일 매핑 테이블
- 주가 윈도우 R1~R5는 거래일 기준, 뉴스 윈도우 N1~N6은 캘린더 기준
- 또는 둘 다 거래일 통일
**우선순위:** 중간 (서강대와 협의 후 표준 동결)

### 12.8 NLI 확률 캘리브레이션
**왜:** 모델 출력 p_support가 0.7이면 실제 70% 신뢰인가? RoBERTa는 일반적으로 over-confident.
**방법:**
- dev 셋에서 reliability diagram 그리기
- Platt scaling 또는 isotonic regression으로 보정
- 보정 후 PTEI 재산출
**근거:** Guo et al. 2017 (calibration of modern neural networks)
**우선순위:** 중간 (PTEI 절댓값 해석에 영향)

### 12.9 라벨러 일치도 향상
**왜:** Cohen's κ≥0.7 목표. 정책 도메인 stance는 어려운 케이스 다수.
**방법:**
- 1차 라벨링 100건 후 κ 점검, 불일치 case 가이드라인 업데이트
- 어려운 케이스 사전 처리: 위기시 공매도금지, 외국 정책 양면성
- 라벨러 1명이 흔들리면 3차 어드져디케이터 추가
**우선순위:** 높음 (전체 품질의 베이스)

### 12.10 재현성 패키지
**왜:** 학술 신뢰도 + 한국언론진흥재단 사후 검증 대비.
**방법:**
- 코드 GitHub 공개 (정책마스터 PII 제거 후)
- 모델 체크포인트 HuggingFace 업로드
- 라벨 데이터 CC-BY 공개
- 재현 가능한 Docker 컨테이너
**우선순위:** 낮음 (사업 종료 후)

### 12.11 우선순위 요약 (12.1~12.10)

| 우선순위 | 항목 | 시기 |
|---|---|---|
| **높음** | 12.1 모델 해석성, 12.4 외생충격 통제, 12.9 라벨러 일치도, **12.12·12.13·12.14 (v2.6 신설)** | 본 분석 사이클 내 |
| **중간** | 12.2 풀텍스트 보강, 12.3 어휘 drift, 12.5 그레이존, 12.7 거래일, 12.8 캘리브레이션 | 1차 결과 후 보강 |
| **낮음** | 12.6 베이지안 풀링, 12.10 재현성 패키지 | 사업 종료 전 |

### 12.12 과적합 방지 — Sanity / Dev / Test 분리 (v2.6 신설)

**왜:** 10건 sanity로 규칙을 case-specific하게 고치면 99,539건 본 분석에서 일반화 실패. Sanity check는 평가 도구가 아니라 디버깅 도구.

**원칙:**
1. **Sanity check (10건)**: 파이프라인 디버깅 전용. 성능 수치 보고 금지. 결과 표는 "디버깅 로그"로만 기록
2. **Dev set (100건)**: 규칙·임계·가설 템플릿 조정에 사용. 정책 50건 × 산업 5 × 유형 ABCDE stratified
3. **Test set (300건)**: **분석 종료 시점까지 절대 건드리지 않음.** 최종 성능 보고는 단 1회 평가
4. 규칙 수정은 case별이 아니라 **일반화 가능한 언어·정책 원칙**에 한해 허용

**금지 패턴:**
- "case04가 틀려서 숫자 토큰 제외" (X, case-specific)
- "한국어 정책 키워드에서 1글자·순수 숫자 토큰은 의미 구분력이 낮아 제외" (O, 일반 원칙)

**핵심 문장 (보고서·기사·신청서):**
> Sanity check는 파이프라인 디버깅 용도이며 성능 평가로 보고하지 않는다. 최종 성능은 사전 분리된 test set(300건)에서 1회 평가로 결정한다.

**라벨링 가이드 상태 (v2.11):**
- [labeling_guide_v1.1.md](labeling_guide_v1.1.md) — 사람 라벨러 2명 본라벨링용 (Dev 100 + Train 확장)
- 파일럿 30건 사람-AI κ: relevance 0.946, stance 0.938 (가이드 작동성 확인용, 성능 보고 아님)
- v1.1 보강: §3.6 동정·일정 기사 룰, §3.7 배경 안정화 조치 룰

**Dev 100 사람-사람 κ 결과 (정식 라벨 신뢰도):**

| 라벨 | κ | 기준 | 결과 |
|---|---:|---:|---|
| relevance | 0.968 | ≥ 0.7 | PASS |
| stance_label (전체) | 0.946 | ≥ 0.7 | PASS |
| stance_label (rel>0 only) | 0.920 | 참고 | PASS |

- 불일치: 5/100건
- 유형: relevance만 차이 2건, stance만 차이 3건, 둘 다 차이 0건
- 정책 편향: pn=24·9·13·50·7 각 1건씩 분산
- 판정: Dev 100 라벨 데이터 채택 가능, 1차 모델 평가 및 PTEI 본 분석 진입 가능

**Zero-shot baseline 결과 (Dev 100 합의 라벨 기준):**

| 평가축 | κ | 해석 |
|---|---:|---|
| relevance 3-class | 0.386 | 0/1/2 구분 약함 |
| relevance binary (rel>0) | 0.346 | 무관/관련 gate도 약함 |
| stance_label 전체 | 0.325 | zero-shot stance 분리 약함 |
| stance_label (rel>0 only) | 0.177 | 정책 관련 기사 stance 판단이 특히 약함 |

- 판정: zero-shot은 1차 탐색용 baseline으로만 사용. 본 결과는 KLUE-RoBERTa 계열 fine-tune 필요성을 지지한다.
- v3.1 relevance gate는 precision 지향으로, Dev 라벨의 배경기사(`relevance=1`)를 `NO_MATCH`로 떨어뜨리는 사례가 많다. fine-tune 후 gate 임계와 L3 배경기사 조건을 재평가한다.
- 산출물: [reports/dev100_zero_shot_eval_summary.md](reports/dev100_zero_shot_eval_summary.md)

### 12.13 Relevance Gate 4줄 원칙 (v2.6 신설)

case별 예외 누적을 막기 위해 gate 규칙을 4줄로 동결한다. 향후 case-by-case 규칙 추가 금지.

```text
L1: 정책명/약칭 normalized exact match           → rel = 1.0
L2: 핵심 정책어 2개 이상 token match            → rel = 0.7
L3: 산업어 + 정책행위어 match                   → rel = 0.4
Fail: 종목명 단독·숫자 단독·1글자 토큰·일반 산업어 단독은 통과 불가 → rel = 0
NLI 진입: rel ≥ 0.4
PTEI: rel을 곱셈 인자로 그대로 사용
```

**일반화 가능한 언어 원칙 (case-specific 아닌 정당화):**
- 순수 숫자·1글자 토큰: "한국어 정책 키워드에서 의미 구분력이 낮음"
- 종목명 단독: "정책 관련성의 충분조건이 아니라 specificity 신호. relevance gate가 아닌 PTEI의 specificity factor로 분리"
- L3는 boost가 아닌 통과 조건: "정책명·산업어·정책행위어가 함께 등장해야 정책 직접 기사로 인정"

### 12.13.A channel_match Gate — Hybrid 정의 (v2.7.1 신설)

C-1 sanity(§12.15)에서 zero-shot NLI가 채널 분리를 못 하는 한계가 확인됨에 따라, channel_match gate를 키워드 + 의미 hybrid로 강화. 단 semantic 단독 활성화는 금지 (false positive 차단).

```text
channel_match_{a,c} ≥ τ_c   ⇔
    keyword_hit(a, c)   = True            ────── (1) 키워드 매칭 (channel_keywords.json)
  OR (
      semantic_hit(a, c) = cosine(emb_a, emb_seed_c) ≥ τ_sem
      AND relevance_a   ≥ 0.4              ─── (2) 정책 관련성 선행 조건
      AND policy_keyword_hit_exists       ─── (3) 정책 키워드도 hit
    )
```

**원칙:**
- **(1) 키워드 매칭이 우선** — 가장 안정적, false positive 적음
- **(2) semantic은 keyword와 결합 후에만** — 임베딩 cosine 단독으로 채널을 열지 않음
- **(3) 정책 관련성 선행** — relevance < 0.4면 어떤 채널도 활성화 불가

semantic_hit 임베딩은 KLUE-RoBERTa-large 또는 TWICE 한국어 금융 임베딩 사용. 채널 시드 임베딩은 채널별 키워드 평균 풀링.

### 12.14 가설 템플릿 — 정책 유형 × 채널 매트릭스 (v2.6 신설)

case별 가설 작성 금지. **정책 유형(ABCDE) × 활성 채널 ON/OFF**로 자동 생성하는 일반화 템플릿만 허용.

| 유형 | 호재(+) 템플릿 | 악재(−) 템플릿 | 주 활성 채널 |
|---|---|---|---|
| **A 세제** | "{정책}이 세부담 감소(C1)와 자금 여력 개선(C5)을 통해 {업종} 주가 상승" | "{정책}이 세부담 증가로 {업종} 주가 하락" | C1, C5 |
| **B 규제강화** | "{정책}이 시장 정상화와 신뢰 회복(C6)을 통해 {업종} 주가 상승" | "{정책}이 마진 악화(C2)와 정책 불확실성(C4)을 통해 {업종} 주가 하락" | C2, C4, C6 |
| **B 시장안정형** *(공매도금지·금융안정 등 위기시 B+1)* | "{정책}이 시장 안정·유동성 회복(C5)·투자심리 개선(C6)을 통해 {업종} 주가 상승 압력" | "{정책}이 거래 위축과 시장 왜곡으로 {업종} 주가 하락" | C5, C6 |
| **C 산업육성** | "{정책}이 설비투자 확대(C3)·인프라 구축·시장 신뢰(C6)를 통해 {업종} 주가 상승" | "{정책}이 정책 효과 부진으로 {업종} 주가 하락" | C3, C6 |
| **D 위기 호재** *(위기대응 패키지)* | "{정책}이 유동성 공급(C5)·신뢰 회복(C6)·불확실성 완화(C4)를 통해 {업종} 주가 상승" | (해당 없음 — D 위기 호재면 D_i=+1) | C4, C5, C6 |
| **D 위기 표면화** *(리먼·황우석 폭로)* | (해당 없음 — D 위기 표면화면 D_i=−1) | "{정책}이 패닉·불확실성 확산(C4)·신뢰 훼손(C6)으로 {업종} 주가 하락" | C4, C6 |
| **E 외국 호재** *(IRA 등)* | "{정책}이 수출 수혜(C2)·공급망 우위(C3)를 통해 {업종} 주가 상승" | (해당 없음) | C2, C3 |
| **E 외국 악재** *(미국 수출규제·IRA 축소)* | (해당 없음) | "{정책}이 수출 충격(C2)·공급망 위축(C3)·정책 불확실성(C4)으로 {업종} 주가 하락" | C2, C3, C4 |

### 12.14.A `build_hypothesis(card)` 함수 명세 (v2.7 동결)

§1.4 정책 카드를 입력으로 받아 NLI 후보 라벨 3개를 반환하는 **유일한 가설 생성 경로**. 구현은 [build_hypotheses.py](build_hypotheses.py).

**입력:**
```python
card = {
    'type_subtype': str,        # §1.4.1 8개 중 1개 (필수)
    'D_i_initial': '+' | '-',   # 검증된 시장반응 방향 (필수)
    'industry': str,            # 5개 산업명
    'alias': str,               # 정책 약칭
    'top_stocks': str,          # 대표 종목 (정책마스터에서)
}
```

**출력:**
```python
return (
    support_hypothesis,     # D_i 방향 지지 가설
    contradict_hypothesis,  # D_i 반대 가설
    neutral_hypothesis,     # 무관 가설 (공통 형식)
)
```

**내부 로직 (요지):**
```python
tpl = SUBTYPE_TEMPLATES[card['type_subtype']]   # 16개 템플릿 (8 subtype × pos/neg)
pos = tpl['pos'].format(policy=alias, stocks=stocks, industry=industry)
neg = tpl['neg'].format(policy=alias, stocks=stocks, industry=industry)
neutral = f"이 기사는 '{alias}' 정책과 무관하거나 정책의 주가 영향이 아닌 단순 사실 전달이다"
if D_i == '+':
    return pos, neg, neutral
else:
    return neg, pos, neutral
```

**금지 사항 (§0.3 #8 강제):**
- 정책 i의 가설을 수동으로 작성하지 않는다
- 정책마다 별도 템플릿 변형을 만들지 않는다 (8 subtype × pos/neg = 16개 템플릿만 유지)
- `type_subtype`이 8개로 부족하면 subtype 확장은 가능하나, 정책별 1회성 추가 금지

**type_subtype 결정 (정책 카드 작성 시 1회 동결):**
8개는 §1.4.1 참조. 추가 카테고리가 필요한 경우 본 문서 §1.4.1 표 자체를 업데이트하고 변경이력에 기록.

### 12.15 NLI / channel_match 책임 분리 원칙 (v2.7.1 신설)

#### 배경 — C-1 sanity (2026-05-16) 결과
- 대표 4정책(일본 수출규제·K칩스법 1차·공매도 금지 3차·8·2 대책) × 각 5~6 기사 × 6채널 multi-label NLI 추론
- 4정책 모두에서 dominant channel이 expected와 불일치, 채널 점수 분포 0.2~0.36의 좁은 범위
- 결과는 성능 평가 아닌 디버깅 로그 (§0.3 #6 적용)

#### 도출된 일반 원칙 (case-specific 수정 금지)
1. **Zero-shot 다국어 NLI는 한국어 정책 기사의 세부 전달경로(비용·수요·공급·규제·유동성·심리)를 안정적으로 구분하지 못한다.** 채널 phrase의 미세 의미 차이를 zero-shot으로 학습 못 함
2. **multi-label NLI에서 정책 호재 신호 = 전체 채널 동시 활성화** 경향 (가설 phrase의 stance 동질성). 이는 모델 특성으로 case별 수정으로 해결 안 됨
3. **NLI 단독 채널 분리는 부적합** — fine-tuning 데이터 없이 zero-shot으로 6채널 분리는 모델 한계

#### 책임 분리 원칙 (확정)
```text
NLI가 담당:
  - 각 채널 가설에 대한 stance 확률 (p_support, p_contradict, p_neutral)
  - D_i 방향 지지/반박 판정

channel_match gate가 담당:
  - 기사가 6채널 중 어느 채널에 속하는가의 활성 여부 (binary indicator)
  - 키워드 + 의미 임베딩 hybrid (§12.13.A)
```

**핵심 문장 (대외 표현 표준):**
> 채널별 NLI sanity 결과, zero-shot NLI는 한국어 정책 기사에서 비용·유동성·공급망 같은 세부 전달경로를 안정적으로 구분하지 못했다. 이에 따라 본 방법론은 NLI를 stance 추론에 한정하고, 채널 분리는 사전 정의된 channel_match gate로 수행하도록 책임을 분리한다.

**금지 표현:**
- "C-1이 PTEI 설계를 입증했다" (X — 너무 강함)
- "NLI sanity가 좋은 결과" (X — 사실은 NLI의 한계 확인)

**허용 표현:**
- "C-1이 책임 분리 설계 결정을 정당화/강화했다"
- "NLI 단독 채널 분리가 실패함을 확인했다"

#### 후속 작업 (장기)
- D-2: channel_match hybrid gate sanity (keyword + KLUE 임베딩) — 진행 완료
- 라벨링 후 채널 fine-tuning은 옵션이나 본 분석 사이클에서는 불필요 (gate로 충분 시 패스)

#### D-2 sanity 결과 (2026-05-16)

D-2 sanity에서 hybrid channel gate는 일본 수출규제, K칩스법, 공매도 금지 3개 대표 정책에서 expected channel을 활성화했다. 특히 공매도 금지에서 C5 유동성·C6 투자심리가 활성화되어, NLI 단독 방식보다 정책 전달경로 분리가 개선됨을 확인했다. 단 C4 규제/불확실성 채널은 일반 행정어 과활성 문제가 있어 키워드 정제가 필요하다 (D-4 후속작업, channel_keywords.json v0.2).

이 결과는 §12.15 책임 분리 설계 결정을 데이터로 정당화한다. "입증"이 아닌 "정당화" 톤 유지.

---

## 13. 다음 액션 (우선순위)

| # | 작업 | 담당 | 산출물 | 소요 |
|---|---|---|---|---|
| 1 | Dev 100 zero-shot baseline 평가 | NLP | `reports/dev100_zero_shot_eval_summary.md` | 즉시 |
| 2 | 99,539건 1차 전수 추론 (relevance gate v3.1 + NLI stance + channel_match hybrid gate) | NLP | `data/processed/article_scores_hybrid.csv` | CPU 장시간 |
| 3 | 일별 PTEI 패널 산출 | NLP | `data/processed/daily_ptei_panel.csv` | 전수 추론 직후 |
| 4 | 서강대 CAR 산출물 결합 | NLP + 서강대 | `data/processed/ptei_car_panel.csv` | CAR 수령 후 |
| 5 | H1~H5 검정 및 기사화 표 생성 | NLP + 서강대 | `reports/PTEI_main.md` | 결합 후 |
| 6 | Train 500~1000건 추가 라벨링 및 fine-tune 비교 | 선택 | `splits/train_500_labeling.csv`, T0/T1/T2 비교표 | 2차 보강 |

---

## 14. 데이터 확장 절차

3개 파일 외 추가가 필요한 경우만:
1. 사유 명세 (왜 부족한지)
2. 정책마스터 4_API_파라미터 BigKinds 키로 Tier-B(D±90) 수집
3. CSV/Parquet 추가, 원본 D1 변경 금지
4. §1 표에 D4, D5… 추가 + 갱신일자

---

## 15. 변경 이력

| 버전 | 일자 | 변경 |
|---|---|---|
| v0.1 | 2026-05-16 | 초안 (감성분석 + 부호정렬) |
| v1.0 | 2026-05-16 | 사용자 수정 (Direction Score) |
| v2.0 | 2026-05-16 | PPT-NLP 통합가설 |
| v2.1 | 2026-05-16 | 채널별 원자가설 6개 + 4유형 |
| v2.2 | 2026-05-16 | 시간창 설계 + 2층 검정 |
| v2.3 | 2026-05-16 | 방향 고정 설명모델 + PTEI 음수봉쇄 + 강도·타이밍·위약 검정 |
| v2.4 | 2026-05-16 | PTEI 명명 + 19편 논문 매핑 + 프로젝트 고유 변형 7가지 + 추가 강화 고려사항 11개 |
| v2.5 | 2026-05-16 | 결함 7개 패치: PTEI 풀이 통일·channel_match를 gate로 강등·novelty 이중반영 해소·placebo 2-track 재정의·diffusion 표현 약화 |
| v2.6 | 2026-05-16 | 과적합 방지 원칙 명시(§0.3 #6, §12.12)·relevance gate 4줄 동결(§12.13)·가설 템플릿 정책유형×채널 매트릭스 일반화(§12.14)·sanity/dev/test 분리 |
| v2.7 | 2026-05-16 | §1.4 정책 카드 사양 신설 (type_subtype 8개 동결)·build_hypothesis(card) 함수 명세(§12.14.A)·v0.x/v1.0 분리 원칙(§0.3 #7,#8)·case별 가설 작성 금지 명문화 |
| v2.7.1 | 2026-05-16 | C-1 채널 sanity 결과 반영: NLI=stance·channel_match=채널 책임 분리 명문화(§12.15)·channel_match를 indicator gate `I(·≥τ_c)`로 식 명시(§2.5)·channel_match hybrid 정의(§12.13.A keyword OR semantic with relevance precondition) |
| v2.7.2 | 2026-05-16 | D-2 hybrid gate sanity 결과 §12.15 추가: 일본 수출규제·K칩스법·공매도 금지 3정책에서 expected channel 활성화 확인, 공매도 금지에서 C5/C6 활성화로 표면감성 우회 정당화. C4 일반 행정어 과활성 식별(D-4 보정 예정) |
| v2.8 | 2026-05-16 | policy_cards_v1.0 동결 (50건, 정정 6건 ok 처리)·D_i 3필드 구조(initial/source/final_car) 도입·14_PolicyCard 시트 통합·Dev 100/Test 300/Train 2513 stratified split·Test set 잠금 강제 |
| v2.9 | 2026-05-16 | D-4 channel_keywords v0.2 (C4 일반 행정어 정제, 11개 제거 + polarity 보강)·D-5 subtype 활성 채널 재매핑 (A_tax_support→[C1] 등)·D_crisis_relief vs shock C4 방향 분리 명시·policy_cards_v1.1/policy_hypotheses_v1.1 동결·sanity 재실행 결과 K칩스법 C4 33%p 감소 확인 |
| **v2.10** | **2026-05-16** | **labeling_guide_v1.1 발행 (§3.6 동정·일정 룰·§3.7 배경 안정화 룰 신설)·파일럿 30 사람-AI κ relevance 0.946·stance 0.938 (가이드 작동성 확인)·Dev 100 본라벨링 진입 준비** |
| **v2.11** | **2026-05-16** | **Dev 100 사람-사람 κ PASS(relevance 0.968, stance 0.946, rel>0 stance 0.920)·불일치 5/100건 유형화·라벨링 트랙 완료·zero-shot baseline 약함 확인(relevance κ 0.386, stance rel>0 κ 0.177)·99,539건 PTEI 탐색 분석 진입 및 fine-tune 필요성 명문화** |

---

## 부록 A: 핵심 용어집

| 약어 | 풀이 | 정의 |
|---|---|---|
| PPT-NLP | Policy-to-Price Transmission NLP | 본 방법론 전체 |
| PTEI | **Policy Transmission Evidence Index** | 정책 전달경로 증거 지수 (기사별 채널벡터 ≥ 0) |
| Contradiction | 반박 근거 점수 | 위약·robustness 검정용 |
| NP_support | News Pressure (support) | 일별 지지 근거 압력 |
| NP_contradict | News Pressure (contradict) | 일별 반박 근거 압력 |
| D_i | Direction anchor | 검증된 주가 반응 부호 |
| k_i | Reaction window | 사전 고정 반응창 (표준 D+5) |
| H_{i,c} | Atomic hypothesis | 정책×채널 원자 NLI 가설 |
| Card_i | Policy card | 정책별 메타 + 활성채널 + 부호 |
| C1..C6 | Transmission channels | 비용/수요/공급/규제/유동성/심리 |
| NewsFlow_main | 메인 뉴스 지표 | D-7~D+5 NP_support 합 |
| CAR_main | 메인 주가 지표 | D_i × CAR[0, +5] |

---

## 부록 B: 핵심 문장 (모든 산출물에 반복 사용)

> 본 방법론은 뉴스가 주가 방향을 독립적으로 예측했는지 검정하는 모델이 아니라, 검증된 정책별 시장 반응 방향을 기준으로 뉴스 안에서 그 방향의 정책-가격 전달근거가 얼마나 강하게 형성됐는지를 측정하는 설명 모델이다.

> 50개 정책 모두에서 검증된 주가 반응 방향을 설명하는 정책 전달경로가 뉴스 안에서 확인됐다. 다만 그 근거가 형성된 시점은 즉시·선반영·지연·설명보강 4유형으로 나뉜다.

> 모든 정책에는 동일하게 `뉴스 D-7~D+5`, `주가 CAR[0,+5]` 기준을 적용한다. 정책별 최적창 선택은 메인 결과에 사용하지 않는다.
