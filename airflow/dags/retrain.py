from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task


@dag(
    dag_id="retail_search_retraining",
    description="Validate Amazon ESCI data, train/evaluate LambdaMART, quality-gate and promote artifacts",
    schedule="0 3 1 * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["retail-search", "learning-to-rank"],
)
def retail_search_retraining():
    @task
    def ingest_data() -> dict:
        from retail_search.pipeline import ingest_data as run

        return run()

    @task
    def validate_data(_: dict) -> dict:
        from retail_search.pipeline import validate_and_prepare_data

        return validate_and_prepare_data()

    @task
    def build_splits(_: dict) -> dict:
        from retail_search.pipeline import build_splits as run

        return run()

    @task
    def build_embeddings_index(_: dict) -> dict:
        config = json.loads(json.dumps({"config": "configs/retrieval.yaml"}))
        if not Path(config["config"]).exists():
            raise FileNotFoundError(config["config"])
        return config

    @task
    def build_features(_: dict) -> dict:
        from retail_search.ranking.features import RetailFeatureBuilder

        return {"feature_names": RetailFeatureBuilder.FEATURE_NAMES}

    @task
    def train_ranker(_: dict) -> dict:
        from retail_search.pipeline import train_evaluate_and_promote

        benchmark = train_evaluate_and_promote()
        return {
            "model_version": benchmark["model"]["version"],
            "validation": benchmark["validation"],
            "quality_gate": benchmark["quality_gate"],
        }

    @task
    def evaluate_validation(training: dict) -> dict:
        if training["validation"]["reranker_ndcg_at_10"] <= 0:
            raise ValueError("Validation NDCG is invalid")
        return training

    @task
    def quality_gate(training: dict) -> dict:
        if not training["quality_gate"]["passed"]:
            raise ValueError("Candidate model failed the configured frozen-test promotion gate")
        return training

    @task
    def register_or_publish_model(training: dict) -> str:
        pointer = json.loads(Path("artifacts/promoted.json").read_text(encoding="utf-8"))
        if pointer["version"] != training["model_version"]:
            raise ValueError("Promoted artifact pointer does not match the evaluated candidate")
        return pointer["version"]

    @task
    def smoke_test_serving_artifact(_: str) -> str:
        from retail_search.pipeline import smoke_test_serving_artifact as run

        return run()

    downloaded = ingest_data()
    validated = validate_data(downloaded)
    splits = build_splits(validated)
    embeddings = build_embeddings_index(splits)
    features = build_features(embeddings)
    trained = train_ranker(features)
    evaluated = evaluate_validation(trained)
    approved = quality_gate(evaluated)
    promoted = register_or_publish_model(approved)
    smoke_test_serving_artifact(promoted)


retail_search_retraining()
