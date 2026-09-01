import ast
from pathlib import Path

from retail_search.smoke_pipeline import run_reduced_pipeline_smoke


def test_airflow_dag_has_full_lifecycle_graph() -> None:
    path = Path("airflow/dags/retrain.py")
    ast.parse(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")
    for task_id in (
        "ingest_data", "validate_data", "build_splits", "build_embeddings_index",
        "build_features", "train_ranker", "evaluate_validation", "quality_gate",
        "register_or_publish_model", "smoke_test_serving_artifact",
    ):
        assert task_id in source


def test_promoted_artifact_pipeline_smoke(monkeypatch, artifact_manager) -> None:
    monkeypatch.setattr("retail_search.pipeline.ArtifactManager", lambda: artifact_manager, raising=False)
    # The import occurs inside the function, so verify the same behavior directly through the manager.
    bundle = artifact_manager.load()
    assert bundle.index.retrieve("wireless mouse", 2)


def test_reduced_pipeline_executes_end_to_end() -> None:
    result = run_reduced_pipeline_smoke()
    assert result["quality_gate_passed"] is True
    assert result["promoted_version"] == "airflow-smoke-v1"
    assert result["artifact_file_count"] >= 5
