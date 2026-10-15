"""User-side feature transforms: static profile attributes plus identity.

Every transform here returns a ``pd.Series`` keyed by ``user_index`` whose
values are single vocabulary tokens (already prefixed, e.g.
``"skin_type=Oily"``), or ``None`` where the feature does not apply.
"""

from __future__ import annotations

import pandas as pd


def user_skin_type(reviews: pd.DataFrame) -> pd.Series:
    """One token per user from their self-reported skin type.

    Args:
        reviews: The full interaction frame (any partition works: this is a
            static profile attribute, not an aggregate, so it is not subject
            to the train-only rule).

    Returns:
        ``user_index -> "skin_type=<value>"``, first occurrence per user.
    """
    first = reviews.drop_duplicates(subset=["user_index"], keep="first")
    return (
        first.set_index("user_index")["skin_type"]
        .map(lambda value: f"skin_type={value}")
    )


def user_skin_tone(reviews: pd.DataFrame) -> pd.Series:
    """One token per user from their self-reported skin tone."""
    first = reviews.drop_duplicates(subset=["user_index"], keep="first")
    return (
        first.set_index("user_index")["skin_tone"]
        .map(lambda value: f"skin_tone={value}")
    )


def user_identity_token(train: pd.DataFrame) -> pd.Series:
    """Per-user ID token, restricted to users present in ``train``.

    Args:
        train: The training partition. Passing anything else (validation,
            test, the full frame) would leak identity for users who are
            supposed to be cold, which is exactly the bug
            ``tests/test_features_leakage.py`` checks for.

    Returns:
        ``user_index -> "user_id=<index>"`` for users present in ``train``.
    """
    users = pd.Index(train["user_index"].unique())
    return pd.Series(
        {int(u): f"user_id={int(u)}" for u in users},
        dtype=object,
    )
