import pandas as pd
import pytest

from retail_search.data.split import assert_disjoint_queries, assign_query_splits, stable_bucket


def test_official_test_is_frozen_and_hash_split_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2, 3, 3],
            "source_split": ["train", "train", "train", "train", "test", "test"],
        }
    )
    first = assign_query_splits(frame, validation_fraction=0.5, seed=42)
    second = assign_query_splits(frame, validation_fraction=0.5, seed=42)
    assert first.tolist() == second.tolist()
    assert set(first[frame["query_id"] == 3]) == {"test"}
    assert first.groupby(frame["query_id"]).nunique().max() == 1
    assert stable_bucket("query", 42) == stable_bucket("query", 42)


def test_split_assertion_rejects_query_leakage() -> None:
    frame = pd.DataFrame({"query_id": [1, 1], "split": ["train", "test"]})
    with pytest.raises(AssertionError, match="Query leakage"):
        assert_disjoint_queries(frame)
