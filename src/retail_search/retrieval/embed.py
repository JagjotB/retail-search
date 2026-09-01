from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


@dataclass
class LSAEmbedder:
    """A reproducible dense text embedding model based on Latent Semantic Analysis."""

    max_features: int = 40000
    dimensions: int = 96
    min_df: int = 2
    random_state: int = 20260901

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=self.min_df,
                        max_features=self.max_features,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=self.dimensions,
                        n_iter=5,
                        random_state=self.random_state,
                    ),
                ),
                ("normalize", Normalizer(copy=False)),
            ]
        )

    def fit(self, documents: Iterable[str]) -> "LSAEmbedder":
        self.pipeline.fit(list(documents))
        return self

    def encode(self, texts: Sequence[str], batch_size: int = 20000) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        chunks = []
        for start in range(0, len(texts), batch_size):
            chunk = self.pipeline.transform(texts[start : start + batch_size])
            chunks.append(np.asarray(chunk, dtype=np.float32))
        return np.vstack(chunks)

    def pairwise_similarity(self, queries: Sequence[str], documents: Sequence[str]) -> np.ndarray:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have equal length")
        scores = np.empty(len(queries), dtype=np.float32)
        for start in range(0, len(queries), 20000):
            stop = min(start + 20000, len(queries))
            query_vectors = self.encode(list(queries[start:stop]))
            document_vectors = self.encode(list(documents[start:stop]))
            scores[start:stop] = np.einsum("ij,ij->i", query_vectors, document_vectors)
        return scores
