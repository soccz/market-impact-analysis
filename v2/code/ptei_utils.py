import csv
import json
import re
from pathlib import Path


CHANNELS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_or(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [part.strip() for part in re.split(r"\s+OR\s+", text) if part.strip()]


def normalize(text):
    return re.sub(r"\s+", "", (text or "").lower())


def is_valid_token(token):
    return len(token) >= 2 and not token.isdigit()


def policy_key(value):
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def load_policy_keywords(path="50_정책마스터.xlsx"):
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["1_50정책_마스터"]
    policies = {}
    for row in list(ws.iter_rows(values_only=True))[3:]:
        if not row or not row[0]:
            continue
        pn = policy_key(row[0])
        policies[pn] = {
            "industry": row[1] or "",
            "alias": row[3] or "",
            "type": row[6] or "",
            "top_stocks": row[9] or "",
            "kw1": row[10] or "",
            "kw2": row[11] or "",
            "and_cond": row[12] or "",
            "not_words": row[13] or "",
        }
    wb.close()
    return policies


def relevance_v3_1(title, content, policy):
    raw = f"{title or ''} {content or ''}"
    text_norm = normalize(raw)

    for not_word in parse_or(policy.get("not_words", "")):
        if not_word and normalize(not_word) in text_norm:
            return 0.0, {
                "level": "NOT",
                "hit": not_word,
                "specificity_boost": 0.0,
                "stock_hits": [],
                "policy_keyword_hit": False,
            }

    primary = parse_or(policy.get("kw1", ""))
    secondary = parse_or(policy.get("kw2", ""))

    stocks = [
        stock.strip()
        for stock in str(policy.get("top_stocks", "")).split(",")
        if stock.strip() and stock.strip() != "전 종목" and len(stock.strip()) >= 2
    ]
    stock_hits = [stock for stock in stocks if normalize(stock) in text_norm]
    specificity_boost = 0.2 if stock_hits else 0.0

    l1_hits = [kw for kw in primary if kw and normalize(kw) in text_norm]
    if l1_hits:
        return 1.0, {
            "level": "L1_primary_exact",
            "hit": "|".join(l1_hits),
            "specificity_boost": specificity_boost,
            "stock_hits": stock_hits,
            "policy_keyword_hit": True,
        }

    l1_5_hits = [kw for kw in secondary if kw and normalize(kw) in text_norm]
    if l1_5_hits:
        return 0.7, {
            "level": "L1.5_secondary_exact",
            "hit": "|".join(l1_5_hits),
            "specificity_boost": specificity_boost,
            "stock_hits": stock_hits,
            "policy_keyword_hit": True,
        }

    l2_hits = []
    for kw in primary:
        tokens = [tok for tok in re.split(r"[\s·\-]+", kw) if tok]
        valid_tokens = [tok for tok in tokens if is_valid_token(tok)]
        if len(valid_tokens) >= 2 and all(normalize(tok) in text_norm for tok in valid_tokens):
            l2_hits.append(kw)
    if l2_hits:
        return 0.7, {
            "level": "L2_primary_token_AND",
            "hit": "|".join(l2_hits),
            "specificity_boost": specificity_boost,
            "stock_hits": stock_hits,
            "policy_keyword_hit": True,
        }

    l2_5_hits = []
    for kw in secondary:
        tokens = [tok for tok in re.split(r"[\s·\-]+", kw) if tok]
        valid_tokens = [tok for tok in tokens if is_valid_token(tok)]
        if len(valid_tokens) >= 2 and all(normalize(tok) in text_norm for tok in valid_tokens):
            l2_5_hits.append(kw)
    if l2_5_hits:
        return 0.5, {
            "level": "L2.5_secondary_token_AND",
            "hit": "|".join(l2_5_hits),
            "specificity_boost": specificity_boost,
            "stock_hits": stock_hits,
            "policy_keyword_hit": True,
        }

    alias = policy.get("alias", "")
    if alias and normalize(alias) in text_norm:
        return 0.4, {
            "level": "L3_alias",
            "hit": alias,
            "specificity_boost": specificity_boost,
            "stock_hits": stock_hits,
            "policy_keyword_hit": True,
        }

    return 0.0, {
        "level": "NO_MATCH",
        "hit": "",
        "specificity_boost": specificity_boost,
        "stock_hits": stock_hits,
        "policy_keyword_hit": False,
    }


def relevance_label_from_score(score):
    if score >= 0.7:
        return 2
    if score >= 0.4:
        return 1
    return 0


def load_hypotheses(path="policy_hypotheses_v1.1.csv"):
    hypotheses = {}
    for row in load_csv(path):
        pn = policy_key(row["pn"])
        hypotheses[pn] = {
            "support": row["hypothesis_support"],
            "contradict": row["hypothesis_contradict"],
            "neutral": row["hypothesis_neutral"],
        }
    return hypotheses


def score_stance(zsc, text, hypotheses):
    candidates = [hypotheses["support"], hypotheses["contradict"], hypotheses["neutral"]]
    out = zsc(text, candidates, multi_label=False)
    scores = dict(zip(out["labels"], out["scores"]))
    named_scores = {
        "support": float(scores[candidates[0]]),
        "contradict": float(scores[candidates[1]]),
        "neutral": float(scores[candidates[2]]),
    }
    pred = max(named_scores, key=named_scores.get)
    return pred, named_scores


def load_policy_cards(path="policy_cards_v1.1.csv"):
    cards = {}
    for row in load_csv(path):
        pn = policy_key(row["pn"])
        active = [channel for channel in CHANNELS if str(row.get(channel, "")).strip().upper() == "ON"]
        row["active_channels"] = active
        cards[pn] = row
    return cards


def load_channel_keywords(path="channel_keywords_v0.2.json"):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    channel_keywords = {}
    for channel, info in raw["channels"].items():
        keywords = []
        for group in info.get("polarity_keywords", {}).values():
            keywords.extend(group)
        keywords.extend(info.get("neutral_anchors", []))
        channel_keywords[channel] = sorted(set(keywords))
    return channel_keywords


def channel_keyword_hit(text, channel, channel_keywords):
    text_norm = normalize(text)
    for keyword in channel_keywords.get(channel, []):
        if keyword and normalize(keyword) in text_norm:
            return True, keyword
    return False, ""


def row_key(row):
    news_id = row.get("news_id", "").strip()
    if news_id:
        return news_id
    return f"{row.get('pn') or row.get('policy_n')}::{row.get('title', '')[:80]}"
