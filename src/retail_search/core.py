from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    description: str = ""
    brand: str = ""
    category: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)
    locale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    product: Product
    retrieval_score: float


class CatalogAdapter(ABC):
    @abstractmethod
    def normalize_products(self, rows: Iterable[Mapping[str, Any]]) -> list[Product]:
        """Map source-specific rows into the engine product schema."""


class CandidateRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_n: int) -> Sequence[Candidate]:
        """Return candidates ordered by retrieval score."""


class FeatureBuilder(ABC):
    @abstractmethod
    def build(self, query: str, candidates: Sequence[Candidate]) -> Any:
        """Build the same feature schema used by training and serving."""


class Reranker(ABC):
    @abstractmethod
    def score(self, features: Any) -> Sequence[float]:
        """Return one learned score for every candidate."""


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, rows: Any, score_column: str) -> Mapping[str, float]:
        """Evaluate only rows with valid relevance judgments."""
