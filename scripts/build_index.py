from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retail_search.adapters.amazon_esci import AmazonEsciAdapter
from retail_search.artifacts.manager import ArtifactManager
from retail_search.retrieval.index import NumpyVectorIndex


def main() -> None:
    bundle = ArtifactManager().load()
    data = pd.read_parquet("data/processed/esci_us_task1.parquet")
    products = data.drop_duplicates("product_id").sample(
        n=min(12000, data["product_id"].nunique()), random_state=20260901
    )
    products = AmazonEsciAdapter().normalize_products(
        products.rename(columns={"title": "product_title"}).to_dict("records")
    )
    index = NumpyVectorIndex.build(bundle.embedder, products)
    output = Path("artifacts/index/manual")
    index.save(output)
    print(json.dumps({"products": len(products), "path": str(output)}, indent=2))


if __name__ == "__main__":
    main()
