from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from retail_search.core import Candidate, FeatureBuilder

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(TOKEN_PATTERN.findall(str(value).lower()))


def tokenize(value: object) -> list[str]:
    return TOKEN_PATTERN.findall(str(value).lower()) if value is not None else []


@dataclass
class BM25Stats:
    k1: float = 1.5
    b: float = 0.75
    document_count: int = 0
    average_length: float = 1.0
    idf: dict[str, float] = field(default_factory=dict)

    def fit(self, documents: Iterable[str]) -> "BM25Stats":
        frequencies: Counter[str] = Counter()
        total_length = 0
        count = 0
        for document in documents:
            tokens = tokenize(document)
            if not tokens:
                continue
            count += 1
            total_length += len(tokens)
            frequencies.update(set(tokens))
        self.document_count = count
        self.average_length = total_length / max(count, 1)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in frequencies.items()
        }
        return self

    def score(self, query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        counts = Counter(document_tokens)
        length_normalizer = 1 - self.b + self.b * len(document_tokens) / max(self.average_length, 1.0)
        total = 0.0
        for token in set(query_tokens):
            frequency = counts.get(token, 0)
            if frequency:
                total += self.idf.get(token, 0.0) * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * length_normalizer)
                )
        return total


class RetailFeatureBuilder(FeatureBuilder):
    FEATURE_NAMES = [
        "dense_similarity",
        "word_title_tfidf_cosine",
        "char_title_tfidf_cosine",
        "word_richtext_tfidf_cosine",
        "bm25_title",
        "query_coverage",
        "title_coverage",
        "token_jaccard",
        "exact_title_match",
        "phrase_match",
        "prefix_match",
        "all_query_tokens_present",
        "brand_match",
        "color_match",
        "numeric_match",
        "query_token_count",
        "title_token_count",
        "token_length_ratio",
        "character_length_ratio",
        "description_coverage",
        "character_similarity",
        "partial_character_similarity",
        "token_set_similarity",
    ]

    def __init__(self, bm25: BM25Stats | None = None):
        self.bm25 = bm25 or BM25Stats()
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=60000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.cross_encoder = None

    def attach_cross_encoder(self, scorer: object) -> "RetailFeatureBuilder":
        self.cross_encoder = scorer
        return self

    def fit(
        self,
        product_titles: Iterable[str],
        corpus_documents: Iterable[str] | None = None,
    ) -> "RetailFeatureBuilder":
        titles = list(product_titles)
        corpus = list(corpus_documents) if corpus_documents is not None else titles
        self.bm25.fit(titles)
        self.word_vectorizer.fit(corpus)
        self.char_vectorizer.fit(corpus)
        return self

    @staticmethod
    def _paired_cosine(vectorizer: TfidfVectorizer, left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        scores = np.empty(len(left), dtype=np.float32)
        for start in range(0, len(left), 10000):
            stop = min(start + 10000, len(left))
            left_matrix = vectorizer.transform(left[start:stop])
            right_matrix = vectorizer.transform(right[start:stop])
            scores[start:stop] = np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()
        return scores

    def _one(
        self,
        query: object,
        title: object,
        description: object,
        brand: object,
        color: object,
        dense_similarity: float,
        word_title_cosine: float,
        char_title_cosine: float,
        word_richtext_cosine: float,
    ) -> list[float]:
        query_norm = normalize_text(query)
        title_norm = normalize_text(title)
        query_tokens = query_norm.split()
        title_tokens = title_norm.split()
        query_set = set(query_tokens)
        title_set = set(title_tokens)
        intersection = query_set & title_set
        union = query_set | title_set
        brand_tokens = set(tokenize(brand))
        color_tokens = set(tokenize(color))
        query_numbers = set(NUMBER_PATTERN.findall(str(query).lower()))
        title_numbers = set(NUMBER_PATTERN.findall(str(title).lower()))
        description_tokens = set(tokenize(description))
        return [
            float(dense_similarity),
            float(word_title_cosine),
            float(char_title_cosine),
            float(word_richtext_cosine),
            self.bm25.score(query_tokens, title_tokens),
            len(intersection) / max(len(query_set), 1),
            len(intersection) / max(len(title_set), 1),
            len(intersection) / max(len(union), 1),
            float(bool(query_norm) and query_norm == title_norm),
            float(bool(query_norm) and query_norm in title_norm),
            float(bool(query_norm) and title_norm.startswith(query_norm)),
            float(bool(query_set) and query_set.issubset(title_set)),
            float(bool(brand_tokens & query_set)),
            float(bool(color_tokens & query_set)),
            float(not query_numbers or query_numbers.issubset(title_numbers)),
            float(len(query_tokens)),
            float(len(title_tokens)),
            min(len(query_tokens), len(title_tokens)) / max(len(query_tokens), len(title_tokens), 1),
            min(len(query_norm), len(title_norm)) / max(len(query_norm), len(title_norm), 1),
            len(query_set & description_tokens) / max(len(query_set), 1),
            fuzz.ratio(query_norm, title_norm) / 100.0,
            fuzz.partial_ratio(query_norm, title_norm) / 100.0,
            fuzz.token_set_ratio(query_norm, title_norm) / 100.0,
        ]

    def build_frame(
        self,
        frame: pd.DataFrame,
        dense_scores: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        dense = dense_scores if dense_scores is not None else frame["dense_similarity"].to_numpy()
        descriptions = frame.get("product_description", pd.Series("", index=frame.index)).fillna("")
        bullets = frame.get("product_bullet_point", pd.Series("", index=frame.index)).fillna("")
        brands = frame.get("product_brand", pd.Series("", index=frame.index)).fillna("")
        colors = frame.get("product_color", pd.Series("", index=frame.index)).fillna("")
        queries = frame["query"].fillna("").astype(str).tolist()
        titles = frame["title"].fillna("").astype(str).tolist()
        rich_texts = [
            f"{title} {description} {bullet}"
            for title, description, bullet in zip(titles, descriptions, bullets, strict=True)
        ]
        word_title_scores = self._paired_cosine(self.word_vectorizer, queries, titles)
        char_title_scores = self._paired_cosine(self.char_vectorizer, queries, titles)
        word_richtext_scores = self._paired_cosine(self.word_vectorizer, queries, rich_texts)
        values = [
            self._one(
                query,
                title,
                f"{description} {bullet}",
                brand,
                color,
                float(score),
                float(word_title_score),
                float(char_title_score),
                float(word_richtext_score),
            )
            for query, title, description, bullet, brand, color, score, word_title_score, char_title_score, word_richtext_score in zip(
                queries,
                titles,
                descriptions,
                bullets,
                brands,
                colors,
                dense,
                word_title_scores,
                char_title_scores,
                word_richtext_scores,
                strict=True,
            )
        ]
        result = pd.DataFrame(values, columns=self.FEATURE_NAMES, dtype=np.float32, index=frame.index)
        if self.cross_encoder is not None:
            # The train-only pointwise cross-encoder is optimized on the same
            # rich product representation used here, not title text alone.
            result["cross_encoder_rich_score"] = self.cross_encoder.score_pairs(
                queries, rich_texts
            )
        return result

    def build(self, query: str, candidates: Sequence[Candidate]) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "query": query,
                    "title": candidate.product.title,
                    "product_description": candidate.product.description,
                    "product_bullet_point": "",
                    "product_brand": candidate.product.brand,
                    "product_color": candidate.product.attributes.get("color", ""),
                    "dense_similarity": candidate.retrieval_score,
                }
                for candidate in candidates
            ]
        )
        return self.build_frame(frame)
