"""Transport-only resume on a dedicated local server; frozen requests unchanged."""

import argparse
import sys
from urllib.parse import urlparse

import requests
import experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    args, remaining = parser.parse_known_args()
    target = urlparse(args.base)
    if target.scheme != "http" or target.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("This experiment permits a local model server only")
    original = requests.sessions.Session.request

    def route(session, method, url, **kwargs):
        if url.startswith("http://127.0.0.1:11434/"):
            url = args.base.rstrip("/") + url[len("http://127.0.0.1:11434") :]
        return original(session, method, url, **kwargs)

    requests.sessions.Session.request = route
    sys.argv = [sys.argv[0], *remaining]
    try:
        experiment.main()
    finally:
        requests.sessions.Session.request = original


if __name__ == "__main__":
    main()
