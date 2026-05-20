# Train 500 합의 라벨 — 메서드

## 데이터 출처
- 입력: 3명의 독립 라벨러(annotator 1/2/3)가 동일한 가이드 v1.1으로 채점한 500건
- 라벨 컬럼: `relevance` (0/1/2), `stance_label` (support/contradict/neutral), `confidence` (low/medium/high)

## 일치도 (학술 보고용)
- **Pairwise Cohen's κ**
  - 1↔2: relevance 0.825, stance 0.887
  - 1↔3: relevance 0.552, stance 0.751
  - 2↔3: relevance 0.715, stance 0.817
- **Fleiss' κ (3 raters)**
  - relevance **0.698** (substantial)
  - stance **0.818** (almost perfect)

## 합의 도출
- **만장일치 (3:0)**: 333건 (66.6%) — 그대로 채택
- **다수결 (2:1) 자동 채택**: 128건 (25.6%)
- **저자 검토 (low confidence ≥ 2 케이스)**: 39건 (7.8%)
  - 가이드 v1.1 적용해 저자가 직접 최종 라벨 결정
  - 21건은 다수결 결과와 다르게 결정 (textual coherence 우선)
  - 18건은 다수결 결과 유지

## 강제 룰
- `relevance = 0` → `stance_label = neutral`

## agreement_score
- 각 행에 (relevance 일치율 + stance 일치율) / 2 기록
- 1.0 = 만장일치, 0.667 = 2:1, 0.5 = 1:1:1 (실제 1:1:1은 0건)

## 분포 (저자 검토 후)
- relevance: {'0': 83, '1': 91, '2': 326}
- stance: {'contradict': 59, 'neutral': 135, 'support': 306}

## labeler 표기
- 만장일치·다수결 자동: `consensus_majority_vote`
- 저자 검토 21건: `consensus_majority_vote + expert_review (author)`

## 학술 보고 권장 문장
> Three independent human annotators labeled 500 articles using the same
> labeling guide (v1.1). Pairwise Cohen's κ ranged 0.55-0.83 (relevance)
> and 0.75-0.89 (stance). Fleiss' κ was 0.698 (relevance, substantial)
> and 0.818 (stance, almost perfect). Of the 500 articles, 333 (66.6%)
> received unanimous labels and were retained as-is. Among the 167
> disagreement cases, 128 were resolved by majority voting. The remaining
> 39 cases with two or more low-confidence ratings were reviewed by the
> author and final labels were determined according to the labeling
> guide; 21 of these were re-labeled to better reflect textual coherence
> with the policy direction.
