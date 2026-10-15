"""Serving: precomputed top-N publication and the request-time cold-start fallback."""

from __future__ import annotations

from recommender.serving.handler import lambda_handler, parse_request
from recommender.serving.topn import build_topn_payloads, score_new_user, to_dataframe

__all__ = [
    "lambda_handler",
    "parse_request",
    "build_topn_payloads",
    "score_new_user",
    "to_dataframe",
]
