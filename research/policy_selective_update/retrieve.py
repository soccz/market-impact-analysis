"""Fixed character TF-IDF retrieval from complete PDF pages, without annotations."""

from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_CHARS = 500
STRIDE = 400
TOP_K = 3


def chunks(pages, prefix):
    result = []
    for page, text in enumerate(pages, 1):
        for start in range(0, len(text), STRIDE):
            value = text[start : start + CHUNK_CHARS]
            if value.strip():
                result.append(
                    {
                        "id": f"{prefix}P{page:02d}C{start:05d}",
                        "page": page,
                        "start": start,
                        "end": start + len(value),
                        "text": value,
                    }
                )
    return result


def select(claim, candidates):
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False)
    matrix = vectorizer.fit_transform([c["text"] for c in candidates])
    scores = (matrix @ vectorizer.transform([claim]).T).toarray().ravel()
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))[:TOP_K]
    return [{**candidates[i], "retrieval_score": float(scores[i])} for i in order]


def source_covered(selected, location):
    intervals = sorted(
        (c["start"], c["end"]) for c in selected if c["page"] == location["page"]
    )
    cursor = location["start"]
    for start, end in intervals:
        if start > cursor:
            continue
        if end > cursor:
            cursor = end
    return cursor >= location["end"]
