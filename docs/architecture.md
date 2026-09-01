# Architecture

## Core engine and adapters

The engine owns product normalization contracts, dense retrieval, shared query-product features, learned ranking, evaluation, artifact loading, and service composition. `AmazonEsciAdapter` is the only layer that knows Amazon column names or ESCI labels. A future retailer adapter supplies the normalized `Product` schema (`product_id`, title, description, brand, category, attributes, locale) and may pass candidates from an existing search engine through the same reranker.

## Online path

1. `SearchService` receives a validated query and `top_k`.
2. `NumpyVectorIndex` embeds the query once and returns a configurable candidate depth by dense cosine similarity.
3. `RetailFeatureBuilder` calculates the exact feature schema serialized with the promoted bundle.
4. `LightGBMReranker` scores the candidate feature matrix.
5. FastAPI returns ordered products, retrieval/reranker scores, rank movement, timing, and model/index versions.

The compact exact NumPy index is intentional for an offline portfolio demo. `CandidateRetriever` makes FAISS, HNSW, Elasticsearch/OpenSearch, or a retailer's existing candidate service replaceable.

## Benchmark path

ESCI contains judged candidate lists rather than exhaustive catalog labels. The benchmark therefore embeds and ranks every judged product for a held-out query, then reranks that same universe. This prevents an unjudged product from being silently treated as irrelevant. Recall@50/100 is reported and is near-saturated on these shallow lists (measured maximum 188; only 102 queries exceed 50); open-corpus retrieval quality needs a different, explicitly labeled dataset.

## Artifact lifecycle

Every model bundle contains the LSA pipeline, BM25/feature state, LightGBM booster, demo vectors/products, manifest metadata, byte counts, and SHA-256 hashes. Promotion is an atomic replacement of `artifacts/promoted.json`, and loading validates every checksum before serving.

Airflow orchestrates ingest, validation, split freezing, representation/features, training, validation, the frozen-test promotion gate, registration, and artifact smoke testing. The monthly schedule is illustrative and can be triggered manually.
