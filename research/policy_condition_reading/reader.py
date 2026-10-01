"""Exploratory, source-grounded condition reader. No reference-label imports."""

import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from datetime import date
from pathlib import Path

FIELDS = ("age", "birth", "income", "welfare")
QUERIES = {
    "age": "신청자격 지원대상 연령 나이 만 청년 세 이하",
    "birth": "신청자격 연령 주민등록등본 출생자 생년월일",
    "income": "신청자격 소득요건 가구당 기준 중위소득 이하 건강보험료",
    "welfare": "신청 제외 대상자 기초생활수급자 생계 의료 주거 교육 급여 신청 불가",
}
BENEFITS = {
    "생계": "livelihood",
    "의료": "medical",
    "주거": "housing",
    "교육": "education",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def read_source(path, layout=True):
    """Offsets refer to NFC, whitespace-free text, separately per page/section."""
    path = Path(path)
    if path.read_bytes().startswith(b"%PDF"):
        import fitz

        with fitz.open(path) as doc:
            return [compact(page.get_text(sort=layout)) for page in doc]
    with zipfile.ZipFile(path) as archive:
        sections = sorted(
            n for n in archive.namelist() if re.fullmatch(r"Contents/section\d+.xml", n)
        )
        return [
            compact(
                "".join(
                    e.text or ""
                    for e in ET.fromstring(archive.read(n)).iter()
                    if e.tag.endswith("}t")
                )
            )
            for n in sections
        ]


def chunks(pages, size=320, stride=160):
    return [
        dict(page=p + 1, start=s, end=min(s + size, len(text)), text=text[s : s + size])
        for p, text in enumerate(pages)
        for s in range(0, len(text), stride)
    ]


def evidence(chunk, start, end):
    return dict(
        page=chunk["page"],
        start=chunk["start"] + start,
        end=chunk["start"] + end,
        sha256=sha(chunk["text"][start:end].encode()),
    )


def parse(chunk, field):
    """Same constrained parser for every retrieval method; abstain on no parse."""
    text = chunk["text"]
    candidates = []
    if field == "age":
        for m in re.finditer(r"(\d{2})세?[~∼～](?:만)?(\d{2})세", text):
            lo, hi = map(int, m.groups())
            if 0 < lo <= hi < 100:
                candidates.append(([lo, hi], [evidence(chunk, m.start(), m.end())]))
    elif field == "birth":
        pattern = r"(19\d{2}|20\d{2})[.년](\d{1,2})[.월](\d{1,2})[.일]?[~∼～](19\d{2}|20\d{2})[.년](\d{1,2})[.월](\d{1,2})[.일]?"
        for m in re.finditer(pattern, text):
            context = text[max(0, m.start() - 50) : m.end() + 20]
            if not any(s in context for s in ("출생", "연령", "주민등록등본상")):
                continue
            try:
                value = [
                    date(*map(int, m.groups()[:3])).isoformat(),
                    date(*map(int, m.groups()[3:])).isoformat(),
                ]
            except ValueError:
                continue
            candidates.append((value, [evidence(chunk, m.start(), m.end())]))
    elif field == "income":
        matches = list(re.finditer(r"중위소득(\d+(?:\.\d+)?)%이하", text))
        scoped = {}
        scoped_evidence = []
        for m in matches:
            prefix = text[max(0, m.start() - 30) : m.start()]
            scope = (
                "first"
                if "최초수혜자" in prefix
                else ("returning" if re.search(r"\d{4}년수혜자", prefix) else "all")
            )
            value = float(m.group(1))
            if scope != "all":
                scoped[scope] = value
                scoped_evidence.append(evidence(chunk, max(0, m.start() - 30), m.end()))
            else:
                candidates.append(
                    ({"all": value}, [evidence(chunk, m.start(), m.end())])
                )
        if scoped:
            candidates.insert(0, (scoped, scoped_evidence))
    elif field == "welfare":
        # The matched local clause names basic-benefit recipients. Broad lists of
        # other programmes, selection priorities and allowed benefits are not read.
        pattern = (
            r"(?:국민)?기초생활수급(?:자로|자중|자|을받고있는사람)[^※□ㅇ‣\n]{0,85}"
        )
        for m in re.finditer(pattern, text):
            clause = m.group().split("및차상위")[0]
            clause = re.split(r"[)）]", clause)[0]
            names = sorted(v for k, v in BENEFITS.items() if k in clause)
            if names and ("급여" in clause):
                candidates.append(
                    (names, [evidence(chunk, m.start(), m.start() + len(clause))])
                )
    if not candidates:
        return None
    value, spans = candidates[0]
    return {"value": value, "evidence": spans}


def tfidf_ranks(windows):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
    matrix = vectorizer.fit_transform([w["text"] for w in windows])
    queries = vectorizer.transform([compact(QUERIES[f]) for f in FIELDS])
    scores = (queries @ matrix.T).toarray()
    return {
        f: np.argsort(-scores[i], kind="stable").tolist() for i, f in enumerate(FIELDS)
    }


class Encoder:
    """Frozen MLM, mean pooling; explicitly NOT a retrieval-fine-tuned encoder."""

    def __init__(self, path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval()

    def encode(self, texts):
        torch = self.torch
        result = []
        for start in range(0, len(texts), 8):
            batch = self.tokenizer(
                texts[start : start + 8],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                # Attention-mask pooling includes the model's special tokens.
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                result.append(torch.nn.functional.normalize(pooled, dim=-1))
        return torch.cat(result)

    def ranks(self, windows):
        vectors = self.encode([w["text"] for w in windows])
        queries = self.encode([compact(QUERIES[f]) for f in FIELDS])
        scores = (queries @ vectors.T).numpy()
        import numpy as np

        return {
            f: np.argsort(-scores[i], kind="stable").tolist()
            for i, f in enumerate(FIELDS)
        }


def predict(pages, method, ranks=None):
    windows = chunks(pages)
    output = {}
    for field in FIELDS:
        if method == "document_first":
            candidates = [
                dict(page=i + 1, start=0, end=len(t), text=t)
                for i, t in enumerate(pages)
            ]
        else:
            k = 3 if method == "tfidf_consensus3" else 1
            candidates = [windows[i] for i in ranks[field][:k]]
        parsed = [p for c in candidates if (p := parse(c, field)) is not None]
        if not parsed:
            output[field] = dict(value=None, evidence=[], reason="no_parse")
        elif method == "tfidf_consensus3" and any(
            p["value"] != parsed[0]["value"] for p in parsed
        ):
            output[field] = dict(
                value=None, evidence=[], reason="retrieved_values_conflict"
            )
        else:
            output[field] = {**parsed[0], "reason": "parsed"}
        output[field]["retrieved"] = [
            {k: c[k] for k in ("page", "start", "end")} for c in candidates
        ]
    return output


def execute(field, value, profile):
    """Only the selected gate, never whole-policy eligibility."""
    if value is None:
        return "unknown"
    if field == "birth":
        x = profile.get("birth")
        verdict = None if x is None else value[0] <= x <= value[1]
    elif field == "income":
        x, scope = profile.get("income_percent"), profile.get("recipient")
        threshold = value.get("all", value.get(scope))
        verdict = None if x is None or threshold is None else x <= threshold
    elif field == "welfare":
        x = profile.get("benefits")
        verdict = None if x is None else not set(x).intersection(value)
    else:
        x = profile.get("age")
        verdict = None if x is None else value[0] <= x <= value[1]
    return "unknown" if verdict is None else ("pass" if verdict else "fail")
