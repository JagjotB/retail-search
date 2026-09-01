from __future__ import annotations

from typing import Any, Iterable, Mapping

from retail_search.core import CatalogAdapter, Product

ESCI_GAIN = {"E": 3, "S": 2, "C": 1, "I": 0}
# Official Task-1 trec_eval gains: ndcg.1=0,2=0.01,3=0.1,4=1.
AMAZON_REFERENCE_GAIN = {"E": 1.0, "C": 0.1, "S": 0.01, "I": 0.0}


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


class AmazonEsciAdapter(CatalogAdapter):
    """Keeps Amazon-specific columns and ESCI semantics outside the core engine."""

    def normalize_products(self, rows: Iterable[Mapping[str, Any]]) -> list[Product]:
        products: list[Product] = []
        for row in rows:
            products.append(
                Product(
                    product_id=_text(row.get("product_id")),
                    title=_text(row.get("product_title", row.get("title", ""))),
                    description=" ".join(
                        part
                        for part in (
                            _text(row.get("product_description")),
                            _text(row.get("product_bullet_point")),
                        )
                        if part
                    ),
                    brand=_text(row.get("product_brand", row.get("brand", ""))),
                    attributes={"color": _text(row.get("product_color"))},
                    locale=_text(row.get("product_locale", row.get("locale", ""))),
                )
            )
        return products
