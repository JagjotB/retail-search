from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from retail_search.artifacts.manager import ArtifactManager
from retail_search.core import Product
from retail_search.evaluation.metrics import evaluate_rankings
from retail_search.ranking.features import RetailFeatureBuilder
from retail_search.ranking.predict import LightGBMReranker
from retail_search.ranking.train import train_ranker
from retail_search.retrieval.embed import LSAEmbedder
from retail_search.retrieval.index import NumpyVectorIndex


def _fixture_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_number in range(9):
        query_id = f"smoke-q{query_number}"
        query = f"alpha widget {query_number}"
        candidates = [
            (f"Alpha Widget {query_number}", "ExactCo", 3),
            (f"Alpha Widget {query_number} Accessory", "ExactCo", 2),
            (f"Alpha Tool {query_number}", "ToolCo", 1),
            (f"Unrelated Object {query_number}", "OtherCo", 0),
        ]
        for candidate_number, (title, brand, relevance) in enumerate(candidates):
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "product_id": f"smoke-p{query_number}-{candidate_number}",
                    "title": title,
                    "product_description": f"Description for {title}",
                    "product_bullet_point": "offline smoke fixture",
                    "product_brand": brand,
                    "product_color": "",
                    "relevance_label": relevance,
                    "split": "train" if query_number < 6 else "validation",
                }
            )
    return pd.DataFrame(rows)


def run_reduced_pipeline_smoke() -> dict[str, Any]:
    """Execute the DAG's data-to-promotion lifecycle on a tiny offline fixture."""
    frame = _fixture_rows()
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if set(train["query_id"]) & set(validation["query_id"]):
        raise AssertionError("Reduced pipeline fixture leaked query IDs")

    corpus = frame["query"].tolist() + frame["title"].tolist()
    embedder = LSAEmbedder(max_features=200, dimensions=8, min_df=1, random_state=7)
    embedder.fit(corpus)
    feature_builder = RetailFeatureBuilder().fit(frame["title"].tolist(), corpus)

    split_features: dict[str, pd.DataFrame] = {}
    for name, rows in (("train", train), ("validation", validation)):
        rows["dense_similarity"] = embedder.pairwise_similarity(
            rows["query"].tolist(), rows["title"].tolist()
        )
        split_features[name] = feature_builder.build_frame(
            rows, rows["dense_similarity"].to_numpy()
        )

    trained = train_ranker(
        split_features["train"],
        train["relevance_label"],
        train["query_id"],
        split_features["validation"],
        validation["relevance_label"],
        validation["query_id"],
        {
            "n_estimators": 30,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 0.1,
            "random_state": 7,
        },
    )
    reranker = LightGBMReranker(trained.model.booster_, split_features["train"].columns)
    validation["reranker_score"] = reranker.score(split_features["validation"])
    baseline = evaluate_rankings(validation, "dense_similarity")["ndcg_at_10"]
    candidate = evaluate_rankings(validation, "reranker_score")["ndcg_at_10"]
    quality_gate_passed = candidate + 1e-9 >= baseline
    if not quality_gate_passed:
        raise AssertionError("Reduced pipeline candidate regressed on validation")

    products = [
        Product(
            product_id=str(row.product_id),
            title=str(row.title),
            brand=str(row.product_brand),
            locale="us",
        )
        for row in frame.drop_duplicates("product_id").itertuples(index=False)
    ]
    index = NumpyVectorIndex.build(embedder, products)
    with tempfile.TemporaryDirectory(prefix="retail-search-airflow-smoke-") as temporary:
        artifact_dir = Path(temporary) / "artifacts"
        (artifact_dir / "demo").mkdir(parents=True)
        benchmark = {
            "generated_at": "smoke",
            "validation": {"baseline_ndcg_at_10": baseline, "reranker_ndcg_at_10": candidate},
        }
        (artifact_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
        (artifact_dir / "demo" / "curated_queries.json").write_text("[]", encoding="utf-8")
        manager = ArtifactManager(artifact_dir)
        manager.publish(
            "airflow-smoke-v1",
            embedder,
            feature_builder,
            reranker,
            index,
            {"feature_names": list(split_features["train"].columns)},
            quality_gate_passed=True,
        )
        loaded = manager.load()
        retrieved = loaded.index.retrieve("alpha widget 8", 3)
        if not retrieved:
            raise AssertionError("Reduced pipeline promoted index returned no candidates")
        artifact_file_count = len(loaded.manifest["files"])

    return {
        "rows": len(frame),
        "train_queries": int(train["query_id"].nunique()),
        "validation_queries": int(validation["query_id"].nunique()),
        "baseline_ndcg_at_10": baseline,
        "reranker_ndcg_at_10": candidate,
        "quality_gate_passed": quality_gate_passed,
        "promoted_version": "airflow-smoke-v1",
        "artifact_file_count": artifact_file_count,
    }
