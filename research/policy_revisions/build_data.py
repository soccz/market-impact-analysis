"""AI-authored contrast sets with span provenance, not human-annotated gold."""
import hashlib
import json
import random
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ["correction", "policy_change", "equivalent", "undetermined"]
# Different wording families in each split. Each family occurs with every label.
FRAMES = [
    "{topic}. 종전 금액: {old}. 새 금액: {new}. {reason}",
    "{topic}. 새로 제시한 금액은 {new}이다. 앞서 제시한 금액은 {old}이었다. {reason}",
    "{topic}에 관한 안내. {reason} 이전에는 {old}, 이후에는 {new}으로 안내했다.",
    "{topic}. {reason} 현재 안내 금액 {new}; 과거 안내 금액 {old}.",
    "{topic} 안내에는 다음과 같이 적혀 있다. 이전 표시 {old} / 이후 표시 {new}. {reason}",
    "{topic}의 두 표기를 대조한다. 이후 표시가 {new}이고 이전 표시는 {old}이다. {reason}",
    "{topic} 관련 공지: 최초 표기 {old} → 최종 표기 {new}. {reason}",
    "{topic}의 최종 표기부터 확인하면 {new}이다. 최초 표기에는 {old}이라고 적혀 있다. {reason}",
    "{topic}. {reason} 옛 안내의 금액({old})과 새 안내의 금액({new})을 나란히 제시한다.",
    "{topic}. {reason} 새 안내에는 {new}이라고 썼으며, 옛 안내에는 {old}이라고 썼다.",
]
REASONS = {
    "correction": [
        "앞선 금액에 오기가 있어 정정했다.", "실제 예산 변경이 아니라 잘못 쓴 숫자를 수정한 것이다.",
        "기존 안내의 오류를 바로잡았으며 사업 규모 자체는 바뀌지 않았다.", "담당자가 수치 기재 오류임을 확인하고 정정했다.",
        "이는 종전 공고의 오기를 고친 결과이다.", "재원 변경 없이 잘못 기재한 금액만 바로잡았다.",
        "예산을 새로 바꾼 것이 아니라 최초 안내의 오기임을 밝혀 수정한 결과다.",
        "담당 부서는 최초 표기가 오기였다고 확인했다. 이에 정정한 수치다.",
        "전달 과정에서 잘못 쓴 금액을 바로잡은 것으로, 정책 변경은 아니다.",
        "종전 문서에 수치 오류가 있었으므로 올바른 금액으로 고쳐 적었다.",
    ],
    "policy_change": [
        "이는 오기 정정이 아니라 예산 변경을 확정한 결과이다.", "추가 재원을 편성해 사업 규모를 실제로 변경했다.",
        "기존 숫자는 정확했다. 이후 정책을 바꾸어 금액을 조정하기로 확정했다.", "새 예산을 승인하여 지원 규모가 달라졌다.",
        "사업 규모 조정이 승인되어 예산이 바뀌었다.", "이전 기재에는 오류가 없으며 실제로 지원액을 변경했다.",
        "최초 표기가 틀려서 정정한 것이 아니다. 지원 규모 변경을 최종 의결한 결과다.",
        "사업 조정안이 확정되면서 재원 규모가 달라졌다.",
        "오기를 고치는 작업이 아니라 새 정책 결정에 따라 지급 규모를 바꾼 것이다.",
        "앞선 금액은 당시 정확했지만 이후 재정 편성 변경으로 규모를 조정했다.",
    ],
    "equivalent": [
        "단위만 달리 적었으며 같은 금액이다.", "정정이나 예산 변경 없이 단위만 환산해 표기했다.",
        "표기 단위를 바꾸었을 뿐 실제 금액은 동일하다.", "두 표기는 같은 금액을 다른 단위로 나타낸 것이다.",
        "금액에는 변동이 없고 표시 단위만 다르다.", "단위 환산 후에도 예산 규모는 같다.",
        "두 숫자의 차이는 단위에 따른 것으로 예산 총량은 동일하다.",
        "표현 단위를 달리한 것일 뿐 금전적 규모에는 변화가 없다.",
        "오기 수정도 실제 증감도 없으며 단위만 변환한 동등한 표기다.",
        "숫자 표기 모양은 다르지만 환산하면 동일한 액수다.",
    ],
}
# Ambiguity/negation/conditional/proposal cues appear in every split, with new wording.
UNKNOWN = [
    ["차이가 생긴 이유는 설명되지 않았다.", "정정했다는 보도는 사실이 아니며 새 금액도 확인되지 않았다.", "예산이 변경될 경우를 가정한 예시이며 실제 결정은 아니다.", "지원액 변경안은 검토 중이며 아직 승인하지 않았다."],
    ["두 금액 중 어느 것이 유효한지 알 수 없다.", "오기 정정을 했다는 주장을 부인했으며 제시한 새 금액도 확정하지 않았다.", "아래는 변경을 가정한 시뮬레이션일 뿐 확정 내용이 아니다.", "새 규모는 제안 단계이고 결정된 금액은 아니다."],
    ["정정인지 실제 변경인지 자료만으로 판단할 수 없다.", "이러한 수치로 수정한 적이 없다는 해명이며 변경도 확정되지 않았다.", "예산을 바꾼다면 어떨지 검토하기 위한 가상 수치다.", "정책 조정에 관한 논의일 뿐 확정된 변경은 없다."],
    ["차이의 사유와 유효한 금액은 제시하지 않았다.", "이 금액으로 정정했다는 설명은 잘못됐고 현재액도 미정이다.", "수치 정정을 가정한 교육 자료이며 실제 공지가 아니다.", "금액 변경 여부는 검토 중으로 아직 결론이 없다."],
    ["숫자가 다른 배경과 유효 여부는 적혀 있지 않다.", "정정 완료라는 주장은 부인됐고 새 금액은 확인 전이다.", "오기 수정이 이뤄진다고 가정했을 때의 예시이다.", "지원 규모 조정은 제안일 뿐 최종 승인 전이다."],
    ["금액 차이를 설명할 자료가 없으며 어느 쪽이 맞는지도 모른다.", "수정했다는 말은 사실과 다르며 새 금액은 확정 전이다.", "실제 예산이 아니라 변경 상황을 가정한 숫자이다.", "조정안 심의가 끝나지 않아 실제 변경은 결정되지 않았다."],
    ["최초와 최종이라는 이름만 붙였을 뿐 두 값의 차이와 유효 여부를 밝히지 않았다.", "오기 정정이 있었다는 주장은 부인됐다. 제시한 최종액도 확인되지 않았다.", "실제 공고가 아닌, 수치 오류를 정정할 경우의 가상 사례다.", "예산 변경 방안을 제안했지만 아직 의결되지 않았다."],
    ["둘 중 어느 금액이 맞는지, 왜 다른지 알 수 없다.", "지원액을 바꾸었다는 설명은 사실무근이며 최종액도 미정이다.", "정정이 발생한다면 어떻게 표시할지 보여 주는 연습 자료다.", "추가 재원 편성은 논의 중이며 확정한 바 없다."],
    ["두 기재의 관계나 유효성에 관한 해설은 빠져 있다.", "수치를 고쳐 적었다는 보도는 사실이 아니고 새 금액도 정해지지 않았다.", "정책이 변경된다는 전제 아래 만든 예시이지 결정된 정책이 아니다.", "지원액 조정은 검토 대상이며 승인된 정책이 아니다."],
    ["금액 차이만 보일 뿐 원인과 유효한 쪽은 확인할 수 없다.", "정정한 적이 없다는 해명이다. 현재 금액 역시 확인되지 않은 상태다.", "정정 절차를 설명하려고 만든 가상 공고로 실제 금액을 나타내지 않는다.", "사업 조정안이 제출됐을 뿐 최종 결정은 나지 않았다."],
]


def money(n, style):
    multiplier, unit = [(1, "억원"), (10000, "만 원"), (100000000, "원"), (Decimal(".0001"), "조 원")][style % 4]
    return format(Decimal(n) * multiplier, "f") + unit


def render(split, i, family):
    offset = {"train": 0, "validation": 300, "evaluation": 600}[split]
    a = offset + 20 + i
    b = a + 7 + i % 9
    topic = f"{2031 + offset // 30 + i % 3}년 {split[0].upper()}{i:03d}가상사업의 지원 예산 총액"
    rows = []
    for relation in LABELS:
        new = a if relation == "equivalent" else b
        old_text, new_text = money(a, i), money(new, i + 1)
        reason = UNKNOWN[family][i % 4] if relation == "undetermined" else REASONS[relation][family]
        # Distinct marker tokens are removed before models see the text.
        draft = FRAMES[family].format(topic=topic, old="«OLD»", new="«NEW»", reason=reason)
        distractor = f" 참고로 별도 시설의 운영비는 «OTHER»이다."
        draft = distractor.strip() + " " + draft if i % 2 else draft + distractor
        spans = []
        text = ""
        import re
        values = {"OLD": (old_text, a, "previous"), "NEW": (new_text, new, "current"), "OTHER": (money(a + 91, i + 2), a + 91, "other")}
        last = 0
        for m in re.finditer(r"«(OLD|NEW|OTHER)»", draft):
            text += draft[last:m.start()]
            surface, amount, role = values[m[1]]
            start = len(text)
            text += surface
            spans.append({"start": start, "end": len(text), "text": surface, "won": str(amount * 100000000), "role": "other" if relation == "undetermined" else role})
            last = m.end()
        text += draft[last:]
        frame = f"{split}-{i:03d}"
        rows.append({"id": hashlib.sha256(text.encode()).hexdigest()[:16], "frame": frame, "semantic_id": frame + "-" + relation,
                     "family": family, "text": text, "relation": relation, "spans": spans,
                     "subtype": ["missing_reason", "denial", "conditional", "proposal"][i % 4] if relation == "undetermined" else relation,
                     "label_status": "AI-authored construction; independent human review pending"})
    return rows


def main():
    manifest = ROOT / "data/manifest.json"
    assert not manifest.exists(), "Already frozen; use a new version."
    plan = json.loads((ROOT / "protocol.json").read_text())["data"]
    result = {}
    all_text = set()
    for split in ["train", "validation", "evaluation"]:
        rows = [r for i in range(plan[split]["frames"]) for f in plan[split]["families"] for r in render(split, i, f)]
        texts = {r["text"] for r in rows}
        assert len(texts) == len(rows) and not texts & all_text
        all_text |= texts
        random.Random(612).shuffle(rows)
        path = ROOT / f"data/{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        result[split] = {"rows": len(rows), "frames": plan[split]["frames"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
