# Final benchmark results

Generated: 2026-09-01T10:00:19.009072+00:00
Frozen split manifest: `f3aa2fc470dfaabd6428e3be4ab62401d8db93412ddd85da6a0343c7fd08f23d`
Configuration hash: `b041ca46ec74ba4e14e95c9536444e69511249adc2f4c4b8523b0478b1a6e3ec`

## Acceptance result: PASS

The frozen held-out comparison measured **11.1832%** relative NDCG@10 improvement. The required minimum is **11.0%**. This value is generated from `artifacts/benchmark.json`; it is not a configured claim.

## Dataset and split

- Official source: Amazon Science `amazon-science/esci-data`, US Task-1 reduced (`small_version == 1`)
- Total valid judgments processed: **601,354**
- Unique queries: **29,844**
- Frozen held-out test: **8,956 queries / 181,701 judgments**
- ESCI gains: E=3, S=2, C=1, I=0
- Query IDs are disjoint across train, validation, and test.

## Frozen-test benchmark

| System | NDCG@10 | Relative vs baseline | MRR@10 | Recall@50 | Recall@100 | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense LSA embedding baseline | 0.739009 | - | 0.922760 | 0.999352 | 1.000000 | 15.84 ms |
| Two-stage + LightGBM LambdaMART | 0.821654 | +11.1832% | 0.957245 | 0.999388 | 1.000000 | 183.55 ms |

Recall@50 is near-saturated and Recall@100 reaches 1.0 on the shallow judged candidate universe (observed maximum: 188 candidates; 102 queries exceed 50). Open-corpus demo retrieval is evaluated separately and unjudged products are never treated as irrelevant.

Separately, the Amazon Task-1 TREC gain convention (E=1, C=0.1, S=0.01, I=0) measures 0.602966 for the embedding baseline and 0.737811 for the reranker. It is not the resume metric above.

## Model

- Dense baseline: TF-IDF word/bigram vectors projected to 96-dimensional LSA embeddings, cosine similarity only.
- Reranker: train-only fine-tuned MiniLM-L4 cross-encoder features plus LightGBM LambdaMART. The iteration count was selected on validation, then the ranker was refit on train+validation before the single frozen-test evaluation.
- Best boosting iteration: 104
- Promoted version: `esci-us-20260901T100019Z-b041ca46`

Top feature importances:
- `all_query_tokens_present`: 0.0000
- `bm25_title`: 0.0112
- `brand_match`: 0.0045
- `char_title_tfidf_cosine`: 0.0214
- `character_length_ratio`: 0.0086
- `character_similarity`: 0.0168
- `color_match`: 0.0012
- `cross_encoder_rich_score`: 0.8033


## Reproduce

```bash
python scripts/tasks.py download
python scripts/tasks.py prepare
python scripts/tasks.py benchmark-full
python scripts/tasks.py acceptance
```

The full benchmark is intentionally separate from the one-command promoted-artifact demo (`docker compose up --build`).
