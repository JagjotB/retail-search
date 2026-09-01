from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from retail_search.core import Candidate, CandidateRetriever, Product
from retail_search.retrieval.embed import LSAEmbedder


class NumpyVectorIndex(CandidateRetriever):
    """Exact cosine index for a compact, dependency-light offline demo catalog."""

    def __init__(self, embedder: LSAEmbedder, products: Sequence[Product], vectors: np.ndarray):
        if len(products) != len(vectors):
            raise ValueError("Product and vector counts differ")
        self.embedder = embedder
        self.products = list(products)
        self.vectors = np.asarray(vectors, dtype=np.float32)

    @classmethod
    def build(cls, embedder: LSAEmbedder, products: Sequence[Product]) -> "NumpyVectorIndex":
        vectors = embedder.encode([product.title for product in products])
        return cls(embedder, products, vectors)

    def retrieve(self, query: str, top_n: int) -> list[Candidate]:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        query_vector = self.embedder.encode([query])[0]
        scores = self.vectors @ query_vector
        count = min(top_n, len(scores))
        if count == len(scores):
            indices = np.argsort(-scores, kind="stable")
        else:
            partial = np.argpartition(-scores, count - 1)[:count]
            indices = partial[np.argsort(-scores[partial], kind="stable")]
        return [Candidate(self.products[int(index)], float(scores[int(index)])) for index in indices]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors, allow_pickle=False)
        (directory / "products.json").write_text(
            json.dumps([product.to_dict() for product in self.products], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path, embedder: LSAEmbedder) -> "NumpyVectorIndex":
        products = [Product(**row) for row in json.loads((directory / "products.json").read_text(encoding="utf-8"))]
        # The demo index is intentionally compact. Loading it into memory avoids
        # Windows file locks during artifact cleanup or immutable repackaging.
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        return cls(embedder, products, vectors)
