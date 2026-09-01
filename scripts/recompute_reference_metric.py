from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from retail_search.adapters.amazon_esci import AMAZON_REFERENCE_GAIN
from retail_search.artifacts.manager import ArtifactManager
from retail_search.evaluation.benchmark import _write_human_reports
from retail_search.evaluation.metrics import evaluate_gain_ndcg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bundle = ArtifactManager().load()
    frame = pd.read_parquet("data/processed/esci_us_task1.parquet")
    rows = frame.loc[frame["split"] == "test"].copy().reset_index(drop=True)
    logging.info("Scoring %s frozen-test judgments", f"{len(rows):,}")
    rows["dense_similarity"] = bundle.embedder.pairwise_similarity(
        rows["query"].astype(str).tolist(), rows["title"].astype(str).tolist()
    )
    features = bundle.feature_builder.build_frame(rows, rows["dense_similarity"].to_numpy())
    rows["reranker_score"] = bundle.reranker.score(features)
    rows["relevance_gain"] = rows["esci_label"].map(AMAZON_REFERENCE_GAIN)
    baseline = evaluate_gain_ndcg(rows, "dense_similarity")
    reranked = evaluate_gain_ndcg(rows, "reranker_score")

    benchmark_path = Path("artifacts/benchmark.json")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["amazon_reference_metric"] = {
        "name": "Amazon Task-1 trec_eval gain NDCG@10 (separately reported)",
        "gain_mapping": AMAZON_REFERENCE_GAIN,
        "embedding_only_ndcg_at_10": baseline["ndcg_at_10"],
        "reranker_ndcg_at_10": reranked["ndcg_at_10"],
    }
    temporary = benchmark_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(benchmark_path)
    experiments = json.loads(
        Path("experiments/results/validation_experiments.json").read_text(encoding="utf-8")
    )
    _write_human_reports(benchmark, experiments)
    print(json.dumps(benchmark["amazon_reference_metric"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
