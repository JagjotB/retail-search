from __future__ import annotations

import argparse

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["model_loaded"] is True
        model = client.get("/model-info")
        model.raise_for_status()
        assert model.json()["benchmark"]["quality_gate_passed"] is True
        search = client.post("/search", json={"query": "wireless gaming mouse", "top_k": 5})
        search.raise_for_status()
        payload = search.json()
        assert payload["query"] == "wireless gaming mouse"
        assert len(payload["results"]) == 5
        assert set(payload["timing_ms"]) == {"retrieval", "reranking", "total"}
    print("PASS: /health, /model-info, and /search contracts validated")


if __name__ == "__main__":
    main()
