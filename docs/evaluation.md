# Evaluation contract

## Primary metric

For each frozen test query, candidates are ordered by a system score and graded DCG@10 is divided by ideal DCG@10. Gains use `2^relevance - 1`, with ESCI mapped E=3, S=2, C=1, I=0. Queries with no positive gain receive NDCG 0. The reported value is the unweighted mean over test queries.

```text
relative_gain = (reranker_ndcg10 - embedding_baseline_ndcg10) / embedding_baseline_ndcg10
promotion passes iff relative_gain >= 0.11
```

Baseline and reranker operate on identical judged rows. The embedding baseline uses only normalized LSA cosine similarity; it cannot access lexical features, labels, learned model scores, or position-derived signals.

## Splits and leakage controls

- Amazon's `split == test` query IDs are frozen as test.
- Amazon training query IDs are hashed with a versioned seed into train and validation.
- Split manifests enumerate every query ID and are SHA-256 hashed into the benchmark artifact.
- LSA vocabulary/projection and BM25 document statistics fit only on the training partition.
- LambdaMART is grouped by query and selected with validation NDCG@10.
- The LambdaMART iteration is selected on validation and then refit on train+validation; the fine-tuned cross-encoder uses train only.
- The test split is evaluated only after model/feature selection and final non-test refit.

## Secondary metrics

The benchmark records MRR@10, judged-universe Recall@50/100, p50/p95 CPU scoring latency on a deterministic sample, index/model sizes via artifact manifests, dataset and held-out counts, label distribution, config hash, split hash, generated timestamp, and Git commit when available. The official Amazon Task-1 TREC convention (relevance positions E=4, C=3, S=2, I=1 with direct gains 1, 0.1, 0.01, 0) is reproduced as a separately named reference NDCG@10 so it cannot be confused with the project's primary resume metric.

## Optimization discipline

Validation experiments include the full model and independently trained no-dense and semantic-plus-length ablations. Failed experiments stay in `experiments/experiment_log.md`. Changing the test set, dropping hard queries after inspection, treating unjudged products as negatives, or printing a configured target as a result is prohibited by tests and the acceptance reporter.
