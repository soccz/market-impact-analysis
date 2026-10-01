"""Reconstruct complete HTML content areas from locally saved source bytes.

No model output or gold answer affects HTML selection or paragraph boundaries.
"""

import re
from bs4 import BeautifulSoup


def paragraphs(raw):
    soup = BeautifulSoup(raw, "html.parser")
    node = soup.select_one("#contents")
    if node is None:
        raise ValueError("missing_content_container")
    for element in node.select("script,style"):
        element.decompose()
    # Do not decompose form/input: malformed Work24 HTML nests substantive text
    # beneath an input in html.parser. get_text retains the actual body instead.
    lines = [
        re.sub(r"\s+", " ", t).strip()
        for t in node.get_text("\n", strip=True).splitlines()
    ]
    text = "\n".join(t for t in lines if t)
    chunks, chunk = [], []
    for line in text.splitlines():
        if chunk and len("\n".join(chunk + [line])) > 420:
            chunks.append("\n".join(chunk))
            chunk = []
        chunk.append(line)
    if chunk:
        chunks.append("\n".join(chunk))
    return text, {f"p{i+1:02d}": chunk for i, chunk in enumerate(chunks)}
