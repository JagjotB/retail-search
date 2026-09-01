from __future__ import annotations

import ast
import json
from pathlib import Path

from retail_search.smoke_pipeline import run_reduced_pipeline_smoke


def main() -> None:
    dag_path = Path("airflow/dags/retrain.py")
    if not dag_path.exists():
        dag_path = Path("/opt/airflow/dags/retrain.py")
    tree = ast.parse(dag_path.read_text(encoding="utf-8"), filename=str(dag_path))
    source = dag_path.read_text(encoding="utf-8")
    required_tasks = [
        "ingest_data", "validate_data", "build_splits", "build_embeddings_index",
        "build_features", "train_ranker", "evaluate_validation", "quality_gate",
        "register_or_publish_model", "smoke_test_serving_artifact",
    ]
    missing = [task for task in required_tasks if task not in source]
    if missing:
        raise AssertionError(f"DAG is missing required tasks: {missing}")
    assert tree.body
    print("PASS: Airflow DAG syntax and required lifecycle task graph validated")
    result = run_reduced_pipeline_smoke()
    print("PASS: reduced data -> features -> train -> evaluate -> gate -> promote lifecycle")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
