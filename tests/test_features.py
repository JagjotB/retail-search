import numpy as np
import pandas as pd

from retail_search.ranking.features import RetailFeatureBuilder


def test_feature_schema_and_high_signal_matches() -> None:
    builder = RetailFeatureBuilder().fit(["Logitech wireless gaming mouse", "red cotton shirt"])
    frame = pd.DataFrame(
        {
            "query": ["wireless gaming mouse"],
            "title": ["Logitech wireless gaming mouse"],
            "product_description": ["low latency mouse"],
            "product_bullet_point": ["wireless"],
            "product_brand": ["Logitech"],
            "product_color": ["black"],
        }
    )
    features = builder.build_frame(frame, np.array([0.8]))
    assert features.columns.tolist() == builder.FEATURE_NAMES
    assert features.loc[0, "query_coverage"] == 1.0
    assert features.loc[0, "all_query_tokens_present"] == 1.0
    assert features.loc[0, "dense_similarity"] == np.float32(0.8)


def test_training_and_serving_feature_order_is_identical(product_candidates) -> None:
    builder = RetailFeatureBuilder().fit([candidate.product.title for candidate in product_candidates])
    features = builder.build("gaming mouse", product_candidates)
    assert features.columns.tolist() == builder.FEATURE_NAMES
    assert len(features) == len(product_candidates)
