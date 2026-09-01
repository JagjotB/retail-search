from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from retail_search.adapters.amazon_esci import AMAZON_REFERENCE_GAIN, AmazonEsciAdapter
from retail_search.artifacts.manager import ArtifactManager
from retail_search.core import Product
from retail_search.evaluation.metrics import evaluate_gain_ndcg, evaluate_rankings
from retail_search.ranking.cross_encoder import OpenVINOCrossEncoder
from retail_search.ranking.features import RetailFeatureBuilder
from retail_search.ranking.fine_tune_cross_encoder import train_cross_encoder
from retail_search.ranking.predict import LightGBMReranker
from retail_search.ranking.train import train_final_ranker, train_ranker
from retail_search.retrieval.embed import LSAEmbedder
from retail_search.retrieval.index import NumpyVectorIndex

LOGGER = logging.getLogger(__name__)


def _archive_previous_benchmark(artifact_dir: Path) -> None:
    previous = artifact_dir / "benchmark.json"
    if not previous.exists():
        return
    payload = json.loads(previous.read_text(encoding="utf-8"))
    generated = str(payload.get("generated_at", "unknown")).replace(":", "-")
    history = Path("experiments/results/benchmark_history")
    history.mkdir(parents=True, exist_ok=True)
    destination = history / f"benchmark-{generated}.json"
    if not destination.exists():
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_gain(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / baseline if baseline else 0.0


def _config_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fit_documents(frame: pd.DataFrame, maximum: int, seed: int) -> list[str]:
    queries = frame["query"].drop_duplicates().astype(str).tolist()
    remaining = max(maximum - len(queries), 1)
    titles = frame["title"].drop_duplicates()
    if len(titles) > remaining:
        titles = titles.sample(n=remaining, random_state=seed)
    return queries + titles.astype(str).tolist()


def _features_for_split(
    frame: pd.DataFrame,
    split: str,
    embedder: LSAEmbedder,
    feature_builder: RetailFeatureBuilder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame.loc[frame["split"] == split].copy().reset_index(drop=True)
    started = time.perf_counter()
    rows["dense_similarity"] = embedder.pairwise_similarity(
        rows["query"].astype(str).tolist(), rows["title"].astype(str).tolist()
    )
    features = feature_builder.build_frame(rows, rows["dense_similarity"].to_numpy())
    LOGGER.info(
        "Built %s features for %s rows in %.1fs",
        split,
        f"{len(rows):,}",
        time.perf_counter() - started,
    )
    return rows, features


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in metrics.items() if name != "per_query"}


def _measure_latency(
    test_rows: pd.DataFrame,
    embedder: LSAEmbedder,
    feature_builder: RetailFeatureBuilder,
    reranker: LightGBMReranker,
    query_limit: int = 200,
) -> tuple[dict[str, float], dict[str, float]]:
    baseline_times: list[float] = []
    reranking_times: list[float] = []
    total_times: list[float] = []
    for _, rows in list(test_rows.groupby("query_id", sort=False))[:query_limit]:
        started = time.perf_counter()
        dense = embedder.pairwise_similarity(
            rows["query"].astype(str).tolist(), rows["title"].astype(str).tolist()
        )
        np.argsort(-dense, kind="stable")
        retrieved = time.perf_counter()
        features = feature_builder.build_frame(rows.reset_index(drop=True), dense)
        scores = reranker.score(features)
        np.argsort(-scores, kind="stable")
        finished = time.perf_counter()
        baseline_times.append((retrieved - started) * 1000)
        reranking_times.append((finished - retrieved) * 1000)
        total_times.append((finished - started) * 1000)

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
        }

    baseline = summary(baseline_times)
    reranked = summary(total_times)
    reranked["reranking_p50_ms"] = summary(reranking_times)["p50_ms"]
    reranked["reranking_p95_ms"] = summary(reranking_times)["p95_ms"]
    reranked["sample_queries"] = len(total_times)
    baseline["sample_queries"] = len(total_times)
    return baseline, reranked


def _curated_query_ids(test_rows: pd.DataFrame, count: int = 10) -> list[object]:
    eligible = []
    for query_id, rows in test_rows.groupby("query_id", sort=False):
        labels = set(rows["esci_label"])
        if "E" in labels and ("I" in labels or "C" in labels):
            key = hashlib.sha256(f"curated:{query_id}".encode("utf-8")).hexdigest()
            eligible.append((key, query_id))
    return [query_id for _, query_id in sorted(eligible)[:count]]


def _build_curated_payload(test_rows: pd.DataFrame, query_ids: list[object]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for query_id in query_ids:
        rows = test_rows.loc[test_rows["query_id"] == query_id].copy()
        baseline_order = rows.sort_values("dense_similarity", ascending=False, kind="stable")
        reranked_order = rows.sort_values("reranker_score", ascending=False, kind="stable")
        baseline_rank = {product_id: rank for rank, product_id in enumerate(baseline_order["product_id"], 1)}
        reranked_rank = {product_id: rank for rank, product_id in enumerate(reranked_order["product_id"], 1)}

        def side(
            ordered: pd.DataFrame,
            score_column: str,
            baseline_positions: dict[object, int] = baseline_rank,
            reranked_positions: dict[object, int] = reranked_rank,
        ) -> list[dict[str, Any]]:
            result = []
            for rank, row in enumerate(ordered.head(10).itertuples(index=False), 1):
                result.append(
                    {
                        "rank": rank,
                        "product_id": str(row.product_id),
                        "title": str(row.title),
                        "brand": "" if pd.isna(row.product_brand) else str(row.product_brand),
                        "retrieval_score": round(float(row.dense_similarity), 6),
                        "reranker_score": round(float(row.reranker_score), 6),
                        "score": round(float(getattr(row, score_column)), 6),
                        "rank_movement": baseline_positions[row.product_id]
                        - reranked_positions[row.product_id],
                        "relevance_label": str(row.esci_label),
                    }
                )
            return result

        baseline_metrics = evaluate_rankings(rows, "dense_similarity")
        reranker_metrics = evaluate_rankings(rows, "reranker_score")
        payload.append(
            {
                "query_id": str(query_id),
                "query": str(rows.iloc[0]["query"]),
                "baseline_ndcg_at_10": baseline_metrics["ndcg_at_10"],
                "reranker_ndcg_at_10": reranker_metrics["ndcg_at_10"],
                "relative_gain": _relative_gain(
                    reranker_metrics["ndcg_at_10"], baseline_metrics["ndcg_at_10"]
                ),
                "baseline": side(baseline_order, "dense_similarity"),
                "reranked": side(reranked_order, "reranker_score"),
            }
        )
    return payload


def _demo_products(frame: pd.DataFrame, curated_ids: list[object], maximum: int, seed: int) -> list[Product]:
    curated = frame.loc[frame["query_id"].isin(curated_ids)]
    remaining = frame.loc[~frame["product_id"].isin(curated["product_id"])]
    remaining = remaining.drop_duplicates("product_id")
    sample_count = max(0, maximum - curated["product_id"].nunique())
    if len(remaining) > sample_count:
        remaining = remaining.sample(n=sample_count, random_state=seed)
    product_rows = pd.concat([curated, remaining], ignore_index=True).drop_duplicates("product_id")
    records = product_rows.rename(
        columns={
            "title": "product_title",
        }
    ).to_dict("records")
    return AmazonEsciAdapter().normalize_products(records)


def _write_human_reports(benchmark: dict[str, Any], experiments: list[dict[str, Any]]) -> None:
    baseline = benchmark["systems"]["embedding_only_baseline"]
    reranked = benchmark["systems"]["two_stage_lambdamart"]
    dataset = benchmark["dataset"]
    status = "PASS" if benchmark["quality_gate"]["passed"] else "FAIL"
    markdown = f"""# Final benchmark results

Generated: {benchmark['generated_at']}
Frozen split manifest: `{benchmark['split_manifest_hash']}`
Configuration hash: `{benchmark['configuration_hash']}`

## Acceptance result: {status}

The frozen held-out comparison measured **{benchmark['quality_gate']['relative_ndcg_at_10_gain'] * 100:.4f}%** relative NDCG@10 improvement. The required minimum is **{benchmark['quality_gate']['minimum_relative_gain'] * 100:.1f}%**. This value is generated from `artifacts/benchmark.json`; it is not a configured claim.

## Dataset and split

- Official source: Amazon Science `amazon-science/esci-data`, US Task-1 reduced (`small_version == 1`)
- Total valid judgments processed: **{dataset['valid_judgments']:,}**
- Unique queries: **{dataset['unique_queries']:,}**
- Frozen held-out test: **{dataset['splits']['test']['queries']:,} queries / {dataset['splits']['test']['judgments']:,} judgments**
- ESCI gains: E=3, S=2, C=1, I=0
- Query IDs are disjoint across train, validation, and test.

## Frozen-test benchmark

| System | NDCG@10 | Relative vs baseline | MRR@10 | Recall@50 | Recall@100 | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense LSA embedding baseline | {baseline['ndcg_at_10']:.6f} | - | {baseline['mrr_at_10']:.6f} | {baseline['recall_at_50']:.6f} | {baseline['recall_at_100']:.6f} | {baseline['latency']['p95_ms']:.2f} ms |
| Two-stage + LightGBM LambdaMART | {reranked['ndcg_at_10']:.6f} | {benchmark['quality_gate']['relative_ndcg_at_10_gain'] * 100:+.4f}% | {reranked['mrr_at_10']:.6f} | {reranked['recall_at_50']:.6f} | {reranked['recall_at_100']:.6f} | {reranked['latency']['p95_ms']:.2f} ms |

Recall@50 is near-saturated and Recall@100 reaches 1.0 on the shallow judged candidate universe (observed maximum: {dataset['judgments_per_query']['maximum']} candidates; {dataset['judgments_per_query']['queries_over_50']} queries exceed 50). Open-corpus demo retrieval is evaluated separately and unjudged products are never treated as irrelevant.

Separately, the Amazon Task-1 TREC gain convention (E=1, C=0.1, S=0.01, I=0) measures {benchmark['amazon_reference_metric']['embedding_only_ndcg_at_10']:.6f} for the embedding baseline and {benchmark['amazon_reference_metric']['reranker_ndcg_at_10']:.6f} for the reranker. It is not the resume metric above.

## Model

- Dense baseline: TF-IDF word/bigram vectors projected to {benchmark['model']['embedding_dimensions']}-dimensional LSA embeddings, cosine similarity only.
- Reranker: train-only fine-tuned MiniLM-L4 cross-encoder features plus LightGBM LambdaMART. The iteration count was selected on validation, then the ranker was refit on train+validation before the single frozen-test evaluation.
- Best boosting iteration: {benchmark['model']['best_iteration']}
- Promoted version: `{benchmark['model']['version']}`

Top feature importances:
"""
    for name, value in list(benchmark["model"]["feature_importance"].items())[:8]:
        markdown += f"- `{name}`: {value:.4f}\n"
    markdown += """

## Reproduce

```bash
python scripts/tasks.py download
python scripts/tasks.py prepare
python scripts/tasks.py benchmark-full
python scripts/tasks.py acceptance
```

The full benchmark is intentionally separate from the one-command promoted-artifact demo (`docker compose up --build`).
"""
    Path("FINAL_RESULTS.md").write_text(markdown, encoding="utf-8")

    lines = [
        "# Experiment log",
        "",
        "Model and feature choices were made on the validation split only. The frozen test split was evaluated after the configuration was selected.",
        "",
        "| Experiment | Features | Validation NDCG@10 | Relative vs dense baseline |",
        "|---|---|---:|---:|",
    ]
    for experiment in experiments:
        lines.append(
            f"| {experiment['name']} | {experiment['features']} | "
            f"{experiment['validation_ndcg_at_10']:.6f} | {experiment['relative_gain'] * 100:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "Ablations are trained independently. They are diagnostic validation experiments, not alternate test-set attempts.",
        ]
    )
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/experiment_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("experiments/results/validation_experiments.json").write_text(
        json.dumps(experiments, indent=2) + "\n", encoding="utf-8"
    )


def run_full_benchmark(
    data_path: Path = Path("data/processed/esci_us_task1.parquet"),
    artifact_dir: Path = Path("artifacts"),
) -> dict[str, Any]:
    _archive_previous_benchmark(artifact_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed benchmark data not found: {data_path}")
    data_config_path = Path("configs/data.yaml")
    retrieval_config_path = Path("configs/retrieval.yaml")
    ranking_config_path = Path("configs/ranking.yaml")
    retrieval_config = yaml.safe_load(retrieval_config_path.read_text(encoding="utf-8"))
    ranking_config = yaml.safe_load(ranking_config_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(data_path)
    dataset_report = json.loads(Path("data/processed/dataset_report.json").read_text(encoding="utf-8"))
    seed = int(ranking_config["model"]["random_state"])

    embedding_config = retrieval_config["embedding"]
    embedder = LSAEmbedder(
        max_features=int(embedding_config["max_features"]),
        dimensions=int(embedding_config["dimensions"]),
        min_df=int(embedding_config["min_df"]),
        random_state=int(embedding_config["random_state"]),
    )
    train_rows = frame.loc[frame["split"] == "train"]
    LOGGER.info("Fitting LSA embedder on training text only")
    embedder.fit(
        _fit_documents(train_rows, int(embedding_config["max_fit_documents"]), seed)
    )
    unique_train_titles = train_rows["title"].drop_duplicates()
    if len(unique_train_titles) > 300000:
        unique_train_titles = unique_train_titles.sample(n=300000, random_state=seed)
    feature_builder = RetailFeatureBuilder().fit(
        unique_train_titles.astype(str).tolist(),
        _fit_documents(train_rows, 250000, seed),
    )
    cross_encoder_config = ranking_config.get("cross_encoder", {})
    if cross_encoder_config.get("enabled", False):
        model_directory = train_cross_encoder(cross_encoder_config)
        feature_builder.attach_cross_encoder(
            OpenVINOCrossEncoder(
                str(model_directory),
                int(cross_encoder_config["max_length"]),
                int(cross_encoder_config["batch_size"]),
            )
        )

    rows: dict[str, pd.DataFrame] = {}
    features: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        rows[split], features[split] = _features_for_split(frame, split, embedder, feature_builder)

    baseline_validation = evaluate_rankings(rows["validation"], "dense_similarity")
    trained = train_ranker(
        features["train"],
        rows["train"]["relevance_label"],
        rows["train"]["query_id"],
        features["validation"],
        rows["validation"]["relevance_label"],
        rows["validation"]["query_id"],
        ranking_config["model"],
    )
    validation_reranker = LightGBMReranker(trained.model.booster_, features["train"].columns)
    rows["validation"]["reranker_score"] = validation_reranker.score(features["validation"])
    reranked_validation = evaluate_rankings(rows["validation"], "reranker_score")
    experiments = [
        {
            "name": "embedding-only baseline",
            "features": "dense LSA cosine only",
            "validation_ndcg_at_10": baseline_validation["ndcg_at_10"],
            "relative_gain": 0.0,
        },
        {
            "name": "full LambdaMART",
            "features": ", ".join(features["train"].columns),
            "validation_ndcg_at_10": reranked_validation["ndcg_at_10"],
            "relative_gain": _relative_gain(
                reranked_validation["ndcg_at_10"], baseline_validation["ndcg_at_10"]
            ),
        },
    ]
    ablation_parameters = dict(ranking_config["model"])
    ablation_parameters["n_estimators"] = min(180, int(ablation_parameters["n_estimators"]))
    ablation_sets = {
        "ablation: no dense feature": [name for name in features["train"].columns if name != "dense_similarity"],
        "ablation: semantic + length only": [
            "dense_similarity", "query_token_count", "title_token_count",
            "token_length_ratio", "character_length_ratio",
        ],
    }
    for name, columns in ablation_sets.items():
        ablation = train_ranker(
            features["train"][columns],
            rows["train"]["relevance_label"],
            rows["train"]["query_id"],
            features["validation"][columns],
            rows["validation"]["relevance_label"],
            rows["validation"]["query_id"],
            ablation_parameters,
        )
        score = ablation.model.predict(features["validation"][columns], num_iteration=ablation.best_iteration)
        temporary = rows["validation"][["query_id", "relevance_label"]].copy()
        temporary["score"] = score
        metric = evaluate_rankings(temporary, "score")
        experiments.append(
            {
                "name": name,
                "features": ", ".join(columns),
                "validation_ndcg_at_10": metric["ndcg_at_10"],
                "relative_gain": _relative_gain(metric["ndcg_at_10"], baseline_validation["ndcg_at_10"]),
            }
        )

    validation_gain = _relative_gain(reranked_validation["ndcg_at_10"], baseline_validation["ndcg_at_10"])
    if validation_gain < float(ranking_config["quality_gate"]["minimum_relative_gain"]):
        LOGGER.warning("Validation gain %.2f%% is below the target; frozen test will still be reported truthfully", validation_gain * 100)

    # After model/iteration selection on validation, refit the exact selected
    # configuration on all non-test judgments. The official test remains frozen.
    final_features = pd.concat([features["train"], features["validation"]], ignore_index=True)
    final_rows = pd.concat([rows["train"], rows["validation"]], ignore_index=True)
    final_trained = train_final_ranker(
        final_features,
        final_rows["relevance_label"],
        final_rows["query_id"],
        ranking_config["model"],
        trained.best_iteration,
    )
    reranker = LightGBMReranker(final_trained.model.booster_, final_features.columns)
    rows["test"]["reranker_score"] = reranker.score(features["test"])
    baseline_test = evaluate_rankings(rows["test"], "dense_similarity")
    reranked_test = evaluate_rankings(rows["test"], "reranker_score")
    relative_gain = _relative_gain(reranked_test["ndcg_at_10"], baseline_test["ndcg_at_10"])
    minimum_gain = float(ranking_config["quality_gate"]["minimum_relative_gain"])
    quality_passed = relative_gain >= minimum_gain

    reference = rows["test"][["query_id", "esci_label", "dense_similarity", "reranker_score"]].copy()
    reference["relevance_gain"] = reference["esci_label"].map(AMAZON_REFERENCE_GAIN)
    reference_baseline = evaluate_gain_ndcg(reference, "dense_similarity")
    reference_reranked = evaluate_gain_ndcg(reference, "reranker_score")

    baseline_latency, reranked_latency = _measure_latency(
        rows["test"], embedder, feature_builder, reranker
    )
    curated_ids = _curated_query_ids(rows["test"], 10)
    curated_payload = _build_curated_payload(rows["test"], curated_ids)
    demo_products = _demo_products(
        frame,
        curated_ids,
        int(retrieval_config["index"]["demo_catalog_size"]),
        seed,
    )
    index = NumpyVectorIndex.build(embedder, demo_products)
    timestamp = datetime.now(UTC)
    version = f"esci-us-{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{_config_hash(data_config_path, retrieval_config_path, ranking_config_path)[:8]}"
    split_manifest_hash = _config_hash(*sorted(Path("data/processed/splits").glob("*_queries.json")))
    benchmark = {
        "schema_version": "1.0",
        "generated_at": timestamp.isoformat(),
        "git_commit": _git_commit(),
        "configuration_hash": _config_hash(data_config_path, retrieval_config_path, ranking_config_path),
        "split_manifest_hash": split_manifest_hash,
        "dataset": dataset_report,
        "metric_contract": {
            "primary": "mean query NDCG@10",
            "gain_mapping": {"E": 3, "S": 2, "C": 1, "I": 0},
            "candidate_universe": "identical ESCI judged candidates for baseline and reranker",
        },
        "validation": {
            "embedding_only_ndcg_at_10": baseline_validation["ndcg_at_10"],
            "reranker_ndcg_at_10": reranked_validation["ndcg_at_10"],
            "relative_gain": validation_gain,
        },
        "systems": {
            "embedding_only_baseline": {
                **_compact_metrics(baseline_test),
                "latency": baseline_latency,
            },
            "two_stage_lambdamart": {
                **_compact_metrics(reranked_test),
                "latency": reranked_latency,
            },
        },
        "amazon_reference_metric": {
            "name": "Amazon Task-1 trec_eval gain NDCG@10 (separately reported)",
            "gain_mapping": AMAZON_REFERENCE_GAIN,
            "embedding_only_ndcg_at_10": reference_baseline["ndcg_at_10"],
            "reranker_ndcg_at_10": reference_reranked["ndcg_at_10"],
        },
        "quality_gate": {
            "minimum_relative_gain": minimum_gain,
            "relative_ndcg_at_10_gain": relative_gain,
            "passed": quality_passed,
            "claim_eligible": quality_passed and dataset_report["minimum_met"],
        },
        "model": {
            "version": version,
            "ranker": "LightGBM LambdaMART",
            "embedding": "TF-IDF + TruncatedSVD Latent Semantic Analysis",
            "cross_encoder": {
                "repository": cross_encoder_config.get("repository"),
                "revision": cross_encoder_config.get("revision"),
                "runtime": "OpenVINO FP16",
            },
            "embedding_dimensions": int(embedding_config["dimensions"]),
            "best_iteration": trained.best_iteration,
            "final_refit": "train+validation at the validation-selected iteration",
            "feature_names": list(features["train"].columns),
            "feature_importance": final_trained.feature_importance,
            "demo_catalog_products": len(demo_products),
        },
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "demo").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"system": "embedding_only_baseline", **_compact_metrics(baseline_test)},
            {"system": "two_stage_lambdamart", **_compact_metrics(reranked_test)},
        ]
    ).to_csv(artifact_dir / "benchmark.csv", index=False)
    (artifact_dir / "demo" / "curated_queries.json").write_text(
        json.dumps(curated_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if quality_passed:
        ArtifactManager(artifact_dir).publish(
            version,
            embedder,
            feature_builder,
            reranker,
            index,
            {
                "dataset": dataset_report["dataset"],
                "dataset_split_manifest_hash": split_manifest_hash,
                "configuration_hash": benchmark["configuration_hash"],
                "feature_names": list(features["train"].columns),
                "validation_ndcg_at_10": reranked_validation["ndcg_at_10"],
                "test_ndcg_at_10": reranked_test["ndcg_at_10"],
                "relative_gain": relative_gain,
            },
            quality_gate_passed=True,
        )
    _write_human_reports(benchmark, experiments)
    return benchmark
