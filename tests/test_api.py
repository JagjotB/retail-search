from fastapi.testclient import TestClient

from retail_search.api.main import create_app


def test_required_api_contract(artifact_manager) -> None:
    with TestClient(create_app(artifact_manager)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["model_version"] == "fixture-v1"
        search = client.post("/search", json={"query": "wireless gaming mouse", "top_k": 2})
        assert search.status_code == 200
        body = search.json()
        assert body["top_k"] == 2
        assert len(body["results"]) == 2
        assert set(body["timing_ms"]) == {"retrieval", "reranking", "total"}


def test_api_rejects_empty_oversized_and_invalid_top_k(artifact_manager) -> None:
    with TestClient(create_app(artifact_manager)) as client:
        assert client.post("/search", json={"query": "   ", "top_k": 5}).status_code == 422
        assert client.post("/search", json={"query": "x" * 301, "top_k": 5}).status_code == 422
        assert client.post("/search", json={"query": "mouse", "top_k": 0}).status_code == 422
        assert client.post("/search", json={"query": "mouse", "top_k": 51}).status_code == 422


def test_demo_page_and_comparison_use_generated_artifacts(artifact_manager) -> None:
    with TestClient(create_app(artifact_manager)) as client:
        landing = client.get("/")
        assert landing.status_code == 200
        assert "two-stage" in landing.text.lower()
        queries = client.get("/demo/queries").json()
        assert len(queries) == 10
        comparison = client.get("/compare/q1")
        assert comparison.status_code == 200
        assert comparison.json()["reranker_ndcg_at_10"] == 0.6
