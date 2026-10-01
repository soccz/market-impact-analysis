"""Natural-language claims -> versioned caps -> audit and update stored answers.

Source contracts reuse the prior selective-update study. The source and claim
grammars below never receive evaluation labels or reference slots.
"""

import json
import re
from datetime import date
from pathlib import Path
import sys
from bridge import ROOT, money, obj, enum, MODELS, OPTIONS

UPDATE = ROOT.parent / "policy_selective_update"
sys.path.append(str(UPDATE))
from temporal_contract import validate, predict

DATE = r"(?P<y>\d{4})\s*[.년-]\s*(?P<m>\d{1,2})\s*[.월-]\s*(?P<d>\d{1,2})\s*[.일]?"
AMOUNT = r"\d[\d,.조억만천백십\s]*원"


def iso(match):
    return date(int(match["y"]), int(match["m"]), int(match["d"])).isoformat()


def source_contract(evidence):
    """One cap, or new cap after an explicit insurance-join date + old cap."""
    if len(evidence) != 1:
        return None
    key, text = next(iter(evidence.items()))
    caps = list(re.finditer(r"최대\s*(" + AMOUNT + r")", text))
    dates = list(re.finditer(DATE + r"\s*이후\s*보증\s*가입", text))
    if len(caps) == 1 and not re.search(r"\d{4}|이전|이후|예외|다만", text):
        return dict(
            branches=[
                dict(
                    from_inclusive=None,
                    until_exclusive=None,
                    max_krw=money(caps[0][1]),
                    evidence=[key],
                )
            ]
        )
    if (
        len(caps) == 2
        and len(dates) == 1
        and re.search(r"이전\s*보증\s*가입자는\s*최대", text)
    ):
        boundary = iso(dates[0])
        return dict(
            branches=[
                dict(
                    from_inclusive=None,
                    until_exclusive=boundary,
                    max_krw=money(caps[1][1]),
                    evidence=[key],
                ),
                dict(
                    from_inclusive=boundary,
                    until_exclusive=None,
                    max_krw=money(caps[0][1]),
                    evidence=[key],
                ),
            ]
        )
    return None


def ground_contract(evidence, proposed):
    grounded = source_contract(evidence)
    prior = validate(proposed, list(evidence))
    target = validate(grounded, list(evidence))
    if target is None:
        return None, dict(action="abstain", reason="source_grammar_not_covered")
    return target, dict(
        action="unchanged" if prior == target else "repaired",
        reason="explicit_source_cap_and_date_spans",
        source_contract=grounded,
    )


def claim_slots(claim):
    dates = list(re.finditer(r"(?:내|나의|본인의)\s*보증\s*가입일은\s*" + DATE, claim))
    caps = list(
        re.finditer(r"(?:이|내)\s*보증료\s*지원\s*상한은\s*(" + AMOUNT + r")", claim)
    )
    unknown = bool(
        re.search(
            r"(?:내|나의|본인의)\s*보증\s*가입일은\s*(?:밝히지|모르|알\s*수\s*없)",
            claim,
        )
    )
    try:
        joined = iso(dates[0]) if len(dates) == 1 and not unknown else None
        cap = money(caps[0][1]) if len(caps) == 1 else None
    except ValueError:
        return dict(status="unsupported", joined_on=None, asserted_cap=None)
    return dict(
        status=(
            "ready"
            if cap is not None
            and (len(dates) == 1 and not unknown or not dates and unknown)
            else "unsupported"
        ),
        joined_on=joined,
        asserted_cap=cap,
    )


def slot_request(claim, model):
    prompt = "한국어 주장만 읽고 신청자 본인의 보증 가입일(joined_on)과 주장한 보증료 지원 상한(asserted_cap)을 추출한다. 신청일·계약일·부모의 가입일을 본인의 보증 가입일로 바꾸지 않는다. 미상 가입일은 null, 금액은 원 단위 정수다. 지원 자격을 판단하지 않는다. 날짜는 YYYY-MM-DD. 필요한 사실의 의미를 표현할 수 없으면 status=unsupported다. 가입일이 명시적으로 미상인 것은 ready이며 joined_on=null이다."
    req = dict(
        model=MODELS[model],
        messages=[
            dict(role="system", content=prompt),
            dict(role="user", content=claim),
        ],
        options=dict(OPTIONS, num_predict=512),
        format=obj(
            dict(
                status=enum(["ready", "unsupported"]),
                joined_on={"type": ["string", "null"]},
                asserted_cap={"type": ["integer", "null"]},
            )
        ),
        stream=False,
        keep_alive="5m",
    )
    if model == "qwen":
        req["think"] = False
    return req


def direct_request(evidence, claim, model):
    prompt = "주어진 정책 근거와 주장만으로 보증료 지원 금액 상한에 관한 주장이 맞는지 판정한다. 정확한 상한 금액이 같으면 supported, 다르면 contradicted, 정보가 부족해 확정할 수 없으면 not_established. 보증 가입일과 신청일을 구별한다. JSON만 출력한다."
    req = dict(
        model=MODELS[model],
        messages=[
            dict(role="system", content=prompt),
            dict(
                role="user",
                content=json.dumps(
                    dict(evidence=evidence, claim=claim), ensure_ascii=False
                ),
            ),
        ],
        options=dict(OPTIONS, num_predict=256),
        format=obj(
            dict(decision=enum(["supported", "contradicted", "not_established"]))
        ),
        stream=False,
        keep_alive="5m",
    )
    if model == "qwen":
        req["think"] = False
    return req


def answer(contract, slots):
    if (
        not isinstance(slots, dict)
        or slots.get("status") != "ready"
        or type(slots.get("asserted_cap")) is not int
    ):
        return "abstain"
    return predict(contract, slots.get("joined_on"), slots["asserted_cap"])


def update(before, after, slots, stored):
    old, new = answer(before, slots), answer(after, slots)
    changed = old != new
    prior_wrong = stored != old
    review = changed or prior_wrong or "abstain" in [old, new]
    return dict(
        before=old,
        after=new,
        revision_changes_decision=changed,
        prior_answer_disagrees=prior_wrong,
        review=review,
        stored=stored,
        updated=new if review else stored,
    )
