from __future__ import annotations

import argparse

import httpx
from bs4 import BeautifulSoup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        landing = client.get("/")
        landing.raise_for_status()
        page = BeautifulSoup(landing.text, "html.parser")
        assert "two-stage" in page.get_text(" ").lower()
        assert page.select_one("#baseline-results") is not None
        assert page.select_one("#reranked-results") is not None
        queries = client.get("/demo/queries").json()
        assert len(queries) >= 10
        comparison = client.get(f"/compare/{queries[0]['query_id']}")
        comparison.raise_for_status()
        payload = comparison.json()
        assert payload["baseline"] and payload["reranked"]
        assert "baseline_ndcg_at_10" in payload and "reranker_ndcg_at_10" in payload
    print("PASS: landing page and held-out benchmark comparison flow validated")


if __name__ == "__main__":
    main()
