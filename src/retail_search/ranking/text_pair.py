from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from retail_search.ranking.features import tokenize

SEPARATOR = "\x1f"
PAIR_FEATURE_NAMES = [
    "pair_probability_irrelevant",
    "pair_probability_complement",
    "pair_probability_substitute",
    "pair_probability_exact",
    "pair_expected_relevance",
]


def pair_analyzer(document: str) -> Iterator[str]:
    """Emit signed query/document identities and bounded cross-token interactions."""
    parts = document.split(SEPARATOR)
    query_tokens = tokenize(parts[0])[:10]
    title_tokens = tokenize(parts[1])[:24] if len(parts) > 1 else []
    brand_tokens = tokenize(parts[2])[:6] if len(parts) > 2 else []
    query_unique = list(dict.fromkeys(query_tokens))
    title_unique = list(dict.fromkeys(title_tokens))
    title_set = set(title_unique)
    for token in query_unique:
        yield f"q:{token}"
        yield f"match:{token}" if token in title_set else f"missing:{token}"
    for token in title_unique:
        yield f"d:{token}"
    for token in set(brand_tokens):
        yield f"brand:{token}"
        if token in query_unique:
            yield f"brand_match:{token}"
    for left, right in zip(query_tokens, query_tokens[1:], strict=False):
        yield f"qb:{left}_{right}"
    for left, right in zip(title_tokens, title_tokens[1:], strict=False):
        yield f"db:{left}_{right}"
    for query_token in query_unique:
        for title_token in title_unique:
            yield f"x:{query_token}>{title_token}"


def pair_documents(frame: pd.DataFrame) -> list[str]:
    brands = frame.get("product_brand", pd.Series("", index=frame.index)).fillna("").astype(str)
    return [
        SEPARATOR.join((str(query), str(title), str(brand)))
        for query, title, brand in zip(frame["query"], frame["title"], brands, strict=True)
    ]


@dataclass
class PairTextScorer:
    n_features: int = 2**19
    alpha: float = 0.000002
    max_iter: int = 20
    random_state: int = 20260901

    def __post_init__(self) -> None:
        self.vectorizer = HashingVectorizer(
            analyzer=pair_analyzer,
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
            dtype=np.float32,
        )
        self.classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=1e-4,
            class_weight="balanced",
            random_state=self.random_state,
            average=True,
            n_jobs=-1,
        )

    def fit(self, frame: pd.DataFrame, labels: Iterable[int]) -> "PairTextScorer":
        matrix = self.vectorizer.transform(pair_documents(frame))
        self.classifier.fit(matrix, np.asarray(list(labels), dtype=np.int8))
        return self

    def predict_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = self.vectorizer.transform(pair_documents(frame))
        probabilities = self.classifier.predict_proba(matrix)
        ordered = np.zeros((len(frame), 4), dtype=np.float32)
        for index, label in enumerate(self.classifier.classes_):
            ordered[:, int(label)] = probabilities[:, index]
        values = np.column_stack((ordered, ordered @ np.arange(4, dtype=np.float32)))
        return pd.DataFrame(values, columns=PAIR_FEATURE_NAMES, index=frame.index, dtype=np.float32)
