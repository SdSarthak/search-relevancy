"""Smoke test a *running* API instance.

This is not part of the pytest suite (which is offline and needs no server).
Start the API first, then:

    python scripts/smoke_test_api.py --url http://localhost:5000

Requires the `requests` package (see requirements-dev.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import requests

DEFAULT_URL = "http://localhost:5000"


class SmokeTestFailure(AssertionError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def test_health(base_url: str, timeout: float) -> dict:
    print("health check...")
    response = requests.get(f"{base_url}/health", timeout=timeout)
    _check(response.status_code == 200, f"expected 200, got {response.status_code}")
    payload = response.json()
    _check(payload["status"] == "healthy", f"service reported {payload}")
    print(f"  ok - {payload['num_articles']} articles indexed")
    return payload


def test_info(base_url: str, timeout: float) -> dict:
    print("service info...")
    response = requests.get(f"{base_url}/info", timeout=timeout)
    _check(response.status_code == 200, f"expected 200, got {response.status_code}")
    payload = response.json()
    print(f"  ok - {json.dumps(payload)}")
    return payload


def test_search(base_url: str, timeout: float, query: str, num_results: int) -> dict:
    print(f"search {query!r}...")
    response = requests.post(
        f"{base_url}/search",
        json={"query": query, "num_results": num_results},
        timeout=timeout,
    )
    _check(response.status_code == 200, f"expected 200, got {response.status_code}")
    payload = response.json()
    _check(payload["results"], "search returned no results")
    scores = [hit["relevance_score"] for hit in payload["results"]]
    _check(scores == sorted(scores, reverse=True), "results are not sorted by score")
    top = payload["results"][0]
    print(f"  ok - top hit: {top['title']} ({top['relevance_score']:.4f})")
    return payload


def test_batch_search(base_url: str, timeout: float, queries: List[str]) -> dict:
    print(f"batch search ({len(queries)} queries)...")
    response = requests.post(
        f"{base_url}/search/batch",
        json={"queries": queries, "num_results": 3},
        timeout=timeout,
    )
    _check(response.status_code == 200, f"expected 200, got {response.status_code}")
    payload = response.json()
    _check(
        len(payload["batch_results"]) == len(queries),
        "batch returned the wrong number of result blocks",
    )
    print("  ok")
    return payload


def test_error_handling(base_url: str, timeout: float) -> None:
    print("error handling...")
    cases = [
        ("missing query", requests.post, "/search", {"json": {}}, 400),
        ("short query", requests.post, "/search", {"json": {"query": "a"}}, 400),
        ("empty batch", requests.post, "/search/batch", {"json": {"queries": []}}, 400),
        ("unknown route", requests.get, "/definitely-not-a-route", {}, 404),
    ]
    for label, method, path, kwargs, expected in cases:
        response = method(f"{base_url}{path}", timeout=timeout, **kwargs)
        _check(
            response.status_code == expected,
            f"{label}: expected {expected}, got {response.status_code}",
        )
        print(f"  ok - {label}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test a running API.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--query", default="climate change")
    parser.add_argument("--num-results", type=int, default=5)
    args = parser.parse_args(argv)

    base_url = args.url.rstrip("/")
    print(f"target: {base_url}\n")

    try:
        test_health(base_url, args.timeout)
        test_info(base_url, args.timeout)
        test_search(base_url, args.timeout, args.query, args.num_results)
        test_batch_search(
            base_url,
            args.timeout,
            ["artificial intelligence", "climate change", "quarterly earnings"],
        )
        test_error_handling(base_url, args.timeout)
    except requests.exceptions.ConnectionError:
        print(f"\nerror: could not connect to {base_url}", file=sys.stderr)
        print("start the API with 'python -m src.app' first", file=sys.stderr)
        return 2
    except SmokeTestFailure as exc:
        print(f"\nfailed: {exc}", file=sys.stderr)
        return 1

    print("\nall smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
