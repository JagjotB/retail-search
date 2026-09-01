from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable task runner for the retail search project")
    parser.add_argument(
        "task",
        choices=[
            "test", "download", "prepare", "train-cross-encoder", "benchmark-full", "acceptance",
            "serve", "smoke", "demo-test", "airflow-smoke",
        ],
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.task == "test":
        run([sys.executable, "-m", "pytest"])
    elif args.task == "download":
        from retail_search.data.ingest import download_official_dataset

        print(json.dumps(download_official_dataset(), indent=2))
    elif args.task == "prepare":
        from retail_search.data.ingest import prepare_amazon_esci

        print(json.dumps(prepare_amazon_esci(), indent=2))
    elif args.task == "train-cross-encoder":
        run([sys.executable, "scripts/train_cross_encoder.py"])
    elif args.task == "benchmark-full":
        from retail_search.evaluation.benchmark import run_full_benchmark

        result = run_full_benchmark()
        print(json.dumps(result["quality_gate"], indent=2))
        if not result["quality_gate"]["passed"]:
            raise SystemExit(1)
    elif args.task == "acceptance":
        from retail_search.evaluation.acceptance import build_acceptance_report

        report = build_acceptance_report()
        print(json.dumps(report, indent=2))
        if not report["overall_passed"]:
            raise SystemExit(1)
    elif args.task == "serve":
        run([sys.executable, "-m", "uvicorn", "retail_search.api.main:app", "--host", "0.0.0.0", "--port", "8000"])
    elif args.task == "smoke":
        run([sys.executable, "scripts/smoke_test.py", "--base-url", args.base_url])
    elif args.task == "demo-test":
        run([sys.executable, "scripts/demo_test.py", "--base-url", args.base_url])
    elif args.task == "airflow-smoke":
        run([sys.executable, "scripts/airflow_smoke.py"])


if __name__ == "__main__":
    main()
