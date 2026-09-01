from retail_search.retrieval.embed import LSAEmbedder
from retail_search.retrieval.index import NumpyVectorIndex


def test_index_returns_deterministic_top_k(products) -> None:
    embedder = LSAEmbedder(max_features=50, dimensions=2, min_df=1, random_state=7)
    embedder.fit([product.title for product in products] + ["wireless gaming mouse"])
    index = NumpyVectorIndex.build(embedder, products)
    first = index.retrieve("wireless gaming mouse", 2)
    second = index.retrieve("wireless gaming mouse", 2)
    assert [item.product.product_id for item in first] == [item.product.product_id for item in second]
    assert len(first) == 2
    assert first[0].retrieval_score >= first[1].retrieval_score
