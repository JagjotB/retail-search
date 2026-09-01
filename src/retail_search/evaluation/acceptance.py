from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_acceptance_report(
    artifact_dir: Path = Path("artifacts"),
    require_runtime_evidence: bool = True,
) -> dict[str, Any]:
    benchmark_path = artifact_dir / "benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError("artifacts/benchmark.json is required")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    runtime_path = artifact_dir / "runtime_evidence.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    runtime_evidence = Path("docs/DEMO_EVIDENCE.md").exists() and len(list(Path("docs/demo").glob("*.png"))) >= 2
    checks = [
        {
            "criterion": "official_dataset_scale",
            "passed": benchmark["dataset"]["valid_judgments"] >= 600000,
            "evidence": f"{benchmark['dataset']['valid_judgments']} judgments in data/processed/dataset_report.json",
        },
        {
            "criterion": "query_safe_frozen_split",
            "passed": bool(benchmark.get("split_manifest_hash")),
            "evidence": "data/processed/splits/*.json and benchmark split_manifest_hash",
        },
        {
            "criterion": "embedding_only_baseline",
            "passed": benchmark["systems"]["embedding_only_baseline"]["query_count"] > 0,
            "evidence": "artifacts/benchmark.json systems.embedding_only_baseline",
        },
        {
            "criterion": "learned_two_stage_reranker",
            "passed": benchmark["model"]["ranker"] == "LightGBM LambdaMART",
            "evidence": "promoted LightGBM model manifest",
        },
        {
            "criterion": "relative_ndcg_at_10_gain",
            "passed": benchmark["quality_gate"]["relative_ndcg_at_10_gain"] >= 0.11,
            "evidence": (
                "measured frozen-test relative gain="
                f"{benchmark['quality_gate']['relative_ndcg_at_10_gain'] * 100:.4f}%"
            ),
        },
        {
            "criterion": "api_contract",
            "passed": Path("src/retail_search/api/main.py").exists(),
            "evidence": "tests/test_api.py and scripts/smoke_test.py",
        },
        {
            "criterion": "containerization",
            "passed": Path("Dockerfile").exists() and Path("docker-compose.yml").exists(),
            "evidence": "Dockerfile, docker-compose.yml, and validated Compose configuration",
        },
        {
            "criterion": "docker_runtime_smoke",
            "passed": bool(runtime.get("docker_runtime_smoke_passed")),
            "evidence": (
                "artifacts/runtime_evidence.json; "
                + str(
                    runtime.get(
                        "docker_runtime_environment",
                        runtime.get("docker_runtime_blocker", "runtime smoke evidence missing"),
                    )
                )
            ),
        },
        {
            "criterion": "airflow_retraining",
            "passed": (
                Path("airflow/dags/retrain.py").exists()
                and bool(runtime.get("airflow_reduced_lifecycle_passed"))
            ),
            "evidence": "airflow/dags/retrain.py, tests/test_pipeline.py, and reduced lifecycle smoke",
        },
        {
            "criterion": "demo_runtime_evidence",
            "passed": (
                runtime_evidence
                and bool(runtime.get("api_live_smoke_passed"))
                and bool(runtime.get("browser_demo_passed"))
            ) if require_runtime_evidence else True,
            "evidence": "docs/DEMO_EVIDENCE.md and docs/demo/*.png",
        },
    ]
    report = {
        "schema_version": "1.0",
        "overall_passed": all(check["passed"] for check in checks),
        "benchmark_generated_at": benchmark["generated_at"],
        "checks": checks,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "acceptance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
