from __future__ import annotations

import time
from typing import Any

import numpy as np

from retail_search.artifacts.manager import ArtifactBundle


class SearchService:
    """Composes retrieval, feature generation, and learned reranking without HTTP concerns."""

    def __init__(self, bundle: ArtifactBundle):
        self.bundle = bundle

    @property
    def version(self) -> str:
        return self.bundle.version

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        total_started = time.perf_counter()
        retrieval_started = total_started
        candidate_depth = min(max(top_k * 5, 50), len(self.bundle.index.products))
        candidates = list(self.bundle.index.retrieve(query, candidate_depth))
        retrieval_finished = time.perf_counter()

        features = self.bundle.feature_builder.build(query, candidates)
        learned_scores = self.bundle.reranker.score(features)
        baseline_order = np.argsort([-candidate.retrieval_score for candidate in candidates], kind="stable")
        baseline_ranks = {int(index): rank for rank, index in enumerate(baseline_order, 1)}
        learned_order = np.argsort(-learned_scores, kind="stable")[:top_k]
        reranking_finished = time.perf_counter()

        results = []
        for rank, candidate_index in enumerate(learned_order, 1):
            index = int(candidate_index)
            candidate = candidates[index]
            results.append(
                {
                    "product_id": candidate.product.product_id,
                    "title": candidate.product.title,
                    "brand": candidate.product.brand,
                    "score": float(learned_scores[index]),
                    "retrieval_score": float(candidate.retrieval_score),
                    "reranker_score": float(learned_scores[index]),
                    "rank": rank,
                    "rank_movement": baseline_ranks[index] - rank,
                }
            )
        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "timing_ms": {
                "retrieval": (retrieval_finished - retrieval_started) * 1000,
                "reranking": (reranking_finished - retrieval_finished) * 1000,
                "total": (reranking_finished - total_started) * 1000,
            },
            "model_version": self.version,
            "index_version": self.version,
        }

    def model_info(self) -> dict[str, Any]:
        benchmark = self.bundle.benchmark
        return {
            "model_version": self.version,
            "index_version": self.version,
            "model_type": benchmark["model"]["ranker"],
            "embedding_type": benchmark["model"]["embedding"],
            "feature_schema": self.bundle.manifest["metadata"]["feature_names"],
            "dataset": {
                "name": benchmark["dataset"]["dataset"],
                "valid_judgments": benchmark["dataset"]["valid_judgments"],
                "unique_queries": benchmark["dataset"]["unique_queries"],
                "test_judgments": benchmark["dataset"]["splits"]["test"]["judgments"],
                "split_manifest_hash": benchmark["split_manifest_hash"],
            },
            "benchmark": {
                "baseline_ndcg_at_10": benchmark["systems"]["embedding_only_baseline"]["ndcg_at_10"],
                "reranker_ndcg_at_10": benchmark["systems"]["two_stage_lambdamart"]["ndcg_at_10"],
                "relative_gain": benchmark["quality_gate"]["relative_ndcg_at_10_gain"],
                "quality_gate_passed": benchmark["quality_gate"]["passed"],
                "p50_latency_ms": benchmark["systems"]["two_stage_lambdamart"]["latency"]["p50_ms"],
                "p95_latency_ms": benchmark["systems"]["two_stage_lambdamart"]["latency"]["p95_ms"],
            },
        }

    def curated_queries(self) -> list[dict[str, str]]:
        return [
            {"query_id": row["query_id"], "query": row["query"]}
            for row in self.bundle.curated_queries
        ]

    def comparison(self, query_id: str) -> dict[str, Any]:
        for row in self.bundle.curated_queries:
            if row["query_id"] == query_id:
                return row
        raise KeyError(query_id)
