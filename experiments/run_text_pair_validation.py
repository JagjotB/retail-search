from __future__ import annotations

import json
import time

import pandas as pd

from retail_search.evaluation.metrics import evaluate_rankings
from retail_search.ranking.text_pair import PairTextScorer


def main() -> None:
    frame = pd.read_parquet("data/processed/esci_us_task1.parquet")
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    started = time.perf_counter()
    scorer = PairTextScorer().fit(train, train["relevance_label"])
    trained = time.perf_counter()
    features = scorer.predict_features(validation)
    validation["pair_text_score"] = features["pair_expected_relevance"]
    metrics = evaluate_rankings(validation, "pair_text_score")
    result = {
        "experiment": "hashed supervised query-title interaction scorer",
        "split": "validation",
        "metrics": {key: value for key, value in metrics.items() if key != "per_query"},
        "training_seconds": trained - started,
        "scoring_seconds": time.perf_counter() - trained,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
