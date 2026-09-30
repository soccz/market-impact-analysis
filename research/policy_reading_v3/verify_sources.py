"""Verify locally retained full-source hashes without redistributing full pages."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-cache", type=Path, required=True)
    p.add_argument("--new-cache", type=Path, required=True)
    args = p.parse_args()
    checks = []
    for line in (ROOT / "data/official.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["known_source"]:
            path = args.old_cache / (row["id"] + ".txt")
        else:
            path = args.new_cache / (
                parse_qs(urlparse(row["url"]).query)["newsId"][0] + ".txt"
            )
        text = path.read_text()
        assert (
            hashlib.sha256(text.encode()).hexdigest() == row["source_normalized_sha256"]
        )
        assert text[row["excerpt_start"] : row["excerpt_end"]] == row["text"]
        assert len(row["text"].split()) <= 25
        checks.append(
            dict(
                id=row["id"],
                source_hash=row["source_normalized_sha256"],
                excerpt_verified=True,
            )
        )
    (ROOT / "source_verification.json").write_text(
        json.dumps(
            dict(status="PASS", sources=checks, n=len(checks)),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("PASS:", len(checks), "source hashes and exact excerpts")


if __name__ == "__main__":
    main()
