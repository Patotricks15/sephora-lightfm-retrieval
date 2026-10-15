"""Item-side feature transforms: catalogue attributes plus identity.

Catalogue attributes (brand, category, price, ingredients) are known the
moment a product is listed, independent of any review ever being written, so
these transforms read ``catalogue`` (indexed by ``item_index``) rather than
the interaction frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_brand(catalogue: pd.DataFrame) -> pd.Series:
    """One token per item from its brand."""
    return catalogue.set_index("item_index")["brand_name"].map(lambda v: f"brand={v}")


def item_category(catalogue: pd.DataFrame) -> pd.Series:
    """One token per item from its primary category."""
    return catalogue.set_index("item_index")["primary_category"].map(lambda v: f"category={v}")


def item_price_bucket(catalogue: pd.DataFrame, edges: list[float]) -> pd.Series:
    """Discretize price into buckets shared by cheap and prestige products alike.

    Args:
        catalogue: Item metadata indexed by ``item_index``.
        edges: Ascending bucket boundaries, e.g. ``[15, 30, 50, 80]`` produces
            five buckets: ``<15``, ``15-30``, ``30-50``, ``50-80``, ``80+``.
    """
    labels = [f"price_bucket={i}" for i in range(len(edges) + 1)]
    bucket_idx = np.digitize(catalogue["price_usd"].to_numpy(), edges)
    tokens = pd.Series(
        [labels[i] for i in bucket_idx], index=catalogue["item_index"].to_numpy()
    )
    tokens.index.name = "item_index"
    return tokens


def item_ingredient_tags(catalogue: pd.DataFrame, separator: str = ";") -> pd.Series:
    """Multi-valued tokens: one list of ``ingredient=<tag>`` per item.

    Returns:
        ``item_index -> list[str]``, unlike the single-valued transforms.
    """
    series = catalogue.set_index("item_index")["ingredients"].fillna("")
    return series.map(
        lambda raw: [f"ingredient={tag}" for tag in raw.split(separator) if tag]
    )


def item_identity_token(train: pd.DataFrame) -> pd.Series:
    """Per-item ID token, restricted to items present in ``train``.

    Args:
        train: The training partition only -- see
            :func:`recommender.features.user_features.user_identity_token` for
            why this matters.
    """
    items = pd.Index(train["item_index"].unique())
    return pd.Series(
        {int(i): f"item_id={int(i)}" for i in items},
        dtype=object,
    )
