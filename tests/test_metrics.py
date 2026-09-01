import math

import pandas as pd

from retail_search.evaluation.metrics import (
    dcg_at_k,
    evaluate_gain_ndcg,
    evaluate_rankings,
    ndcg_at_k,
    ndcg_at_k_from_gains,
)


def test_ndcg_is_one_for_ideal_order() -> None:
    assert ndcg_at_k([3, 2, 1, 0], 10) == 1.0


def test_ndcg_known_reversed_order_and_truncation() -> None:
    actual = ndcg_at_k([0, 1, 2, 3], 3)
    expected = dcg_at_k([0, 1, 2], 3) / dcg_at_k([3, 2, 1, 0], 3)
    assert math.isclose(actual, expected)


def test_ndcg_handles_no_relevant_and_short_queries() -> None:
    assert ndcg_at_k([0, 0], 10) == 0.0
    assert ndcg_at_k([2], 10) == 1.0


def test_evaluate_rankings_averages_per_query() -> None:
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "relevance_label": [3, 0, 0, 2],
            "score": [1.0, 0.0, 1.0, 0.0],
        }
    )
    result = evaluate_rankings(frame, "score")
    assert result["query_count"] == 2
    assert result["judgment_count"] == 4
    assert 0 < result["ndcg_at_10"] < 1


def test_amazon_trec_gain_ndcg_uses_direct_gains() -> None:
    assert ndcg_at_k_from_gains([1.0, 0.1, 0.01, 0.0], 10) == 1.0
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1],
            "relevance_gain": [0.0, 0.01, 0.1, 1.0],
            "score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    result = evaluate_gain_ndcg(frame, "score")
    assert 0 < result["ndcg_at_10"] < 1
