from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yaml

from retail_search.evaluation.metrics import evaluate_rankings
from retail_search.ranking.cross_encoder import OpenVINOCrossEncoder, ensure_cross_encoder


def main() -> None:
    config = yaml.safe_load(Path("configs/ranking.yaml").read_text(encoding="utf-8"))["cross_encoder"]
    local_dir = Path(config["local_dir"])
    if not (local_dir / "openvino" / "model.xml").exists():
        ensure_cross_encoder(config["repository"], config["revision"], local_dir)
    validation = pd.read_parquet("data/processed/esci_us_task1.parquet")
    validation = validation.loc[validation["split"] == "validation"].reset_index(drop=True)
    scorer = OpenVINOCrossEncoder(config["local_dir"], config["max_length"], config["batch_size"])
    started = time.perf_counter()
    rich_text = (
        validation["title"].fillna("").astype(str)
        + " "
        + validation["product_brand"].fillna("").astype(str)
        + " "
        + validation["product_bullet_point"].fillna("").astype(str)
        + " "
        + validation["product_description"].fillna("").astype(str)
    ).str.strip()
    validation["cross_encoder_rich_score"] = scorer.score_pairs(
        validation["query"].astype(str).tolist(), rich_text.tolist()
    )
    metrics = evaluate_rankings(validation, "cross_encoder_rich_score")
    result = {
        "experiment": config["repository"],
        "model_revision": config["revision"],
        "split": "validation",
        "metrics": {key: value for key, value in metrics.items() if key != "per_query"},
        "scoring_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
