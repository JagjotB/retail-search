from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retail_search.data.ingest import download_official_dataset, prepare_amazon_esci
from retail_search.evaluation.acceptance import build_acceptance_report
from retail_search.evaluation.benchmark import run_full_benchmark


def ingest_data() -> dict[str, Any]:
    return download_official_dataset()


def validate_and_prepare_data() -> dict[str, Any]:
    return prepare_amazon_esci()


def build_splits() -> dict[str, Any]:
    report_path = Path("data/processed/dataset_report.json")
    if not report_path.exists():
        raise FileNotFoundError("Prepared dataset report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not all((Path("data/processed/splits") / f"{name}_queries.json").exists() for name in ("train", "validation", "test")):
        raise FileNotFoundError("One or more frozen query manifests are missing")
    return report["splits"]


def train_evaluate_and_promote() -> dict[str, Any]:
    return run_full_benchmark()


def quality_gate() -> dict[str, Any]:
    return build_acceptance_report(require_runtime_evidence=False)


def smoke_test_serving_artifact() -> str:
    from retail_search.artifacts.manager import ArtifactManager

    bundle = ArtifactManager().load()
    candidates = bundle.index.retrieve("wireless gaming mouse", 5)
    if not candidates:
        raise RuntimeError("Promoted index returned no candidates")
    return bundle.version
