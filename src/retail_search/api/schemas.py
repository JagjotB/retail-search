from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_must_have_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace characters")
        return value


class SearchResult(BaseModel):
    product_id: str
    title: str
    brand: str = ""
    score: float
    retrieval_score: float
    reranker_score: float
    rank: int
    rank_movement: int


class Timing(BaseModel):
    retrieval: float
    reranking: float
    total: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]
    timing_ms: Timing
    model_version: str
    index_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    index_version: str
    model_type: str
    embedding_type: str
    feature_schema: list[str]
    dataset: dict[str, Any]
    benchmark: dict[str, Any]
