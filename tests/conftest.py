from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from retail_search.artifacts.manager import ArtifactManager
from retail_search.core import Candidate, Product
from retail_search.ranking.features import RetailFeatureBuilder
from retail_search.ranking.predict import LightGBMReranker
from retail_search.retrieval.embed import LSAEmbedder
from retail_search.retrieval.index import NumpyVectorIndex


@pytest.fixture
def products() -> list[Product]:
    return [
        Product("p1", "Wireless gaming mouse", brand="Vertex", locale="us"),
        Product("p2", "USB office mouse", brand="Plain", locale="us"),
        Product("p3", "Mechanical gaming keyboard", brand="Vertex", locale="us"),
        Product("p4", "Red cotton shirt", brand="Wear", locale="us"),
    ]


@pytest.fixture
def product_candidates(products) -> list[Candidate]:
    return [Candidate(product, 1.0 - index * 0.1) for index, product in enumerate(products)]


@pytest.fixture
def artifact_manager(tmp_path: Path, products) -> ArtifactManager:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    embedder = LSAEmbedder(max_features=50, dimensions=2, min_df=1, random_state=7)
    embedder.fit([product.title for product in products] + ["wireless gaming mouse", "red shirt"])
    index = NumpyVectorIndex.build(embedder, products)
    builder = RetailFeatureBuilder().fit([product.title for product in products])
    candidates = [
        Candidate(product, float(score))
        for product, score in zip(products, [0.9, 0.5, 0.4, 0.1], strict=True)
    ]
    features = builder.build("wireless gaming mouse", candidates)
    training = lgb.Dataset(features, label=np.array([3, 1, 1, 0]))
    booster = lgb.train({"objective": "regression", "verbosity": -1, "seed": 7}, training, num_boost_round=3)
    reranker = LightGBMReranker(booster, features.columns)
    benchmark = {
        "generated_at": "2026-09-01T00:00:00+00:00",
        "split_manifest_hash": "fixture-manifest",
        "dataset": {
            "dataset": "Fixture ESCI",
            "valid_judgments": 600000,
            "unique_queries": 30000,
            "splits": {"test": {"queries": 1, "judgments": 4}},
        },
        "systems": {
            "embedding_only_baseline": {"ndcg_at_10": 0.5},
            "two_stage_lambdamart": {"ndcg_at_10": 0.6, "latency": {"p50_ms": 2.0, "p95_ms": 3.0}},
        },
        "quality_gate": {"relative_ndcg_at_10_gain": 0.2, "passed": True},
        "model": {"ranker": "LightGBM LambdaMART", "embedding": "Fixture LSA"},
    }
    (artifact_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    (artifact_dir / "demo").mkdir()
    comparison = {
        "query_id": "q1",
        "query": "wireless gaming mouse",
        "baseline_ndcg_at_10": 0.5,
        "reranker_ndcg_at_10": 0.6,
        "relative_gain": 0.2,
        "baseline": [{"product_id": "p1"}],
        "reranked": [{"product_id": "p1"}],
    }
    (artifact_dir / "demo" / "curated_queries.json").write_text(
        json.dumps([comparison] * 10), encoding="utf-8"
    )
    manager = ArtifactManager(artifact_dir)
    manager.publish(
        "fixture-v1",
        embedder,
        builder,
        reranker,
        index,
        {"feature_names": list(features.columns)},
        quality_gate_passed=True,
    )
    return manager
