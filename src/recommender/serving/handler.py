"""AWS Lambda entry point serving precomputed top-N recommendations.

Deliberately dependency-free (standard library + boto3, which the Lambda
runtime already provides) so the deployment package stays a single file and
needs no build step or layer.

The ranking itself is not computed here. ``scripts/publish_topn.py`` (see the
docstring in ``serving/topn.py``) writes the lists to DynamoDB after
training, so this handler is a key lookup plus a response envelope for every
user already known at the last training run. A user who is not in the table
gets a 404 with an explicit ``fallback`` hint pointing the caller at the
request-time scoring path documented in ``serving/topn.py::score_new_user``,
rather than silently returning nothing.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any

try:  # pragma: no cover - boto3 is provided by the Lambda runtime.
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None

DEFAULT_K = int(os.environ.get("DEFAULT_K", "10"))
MAX_K = 100
TOPN_TABLE = os.environ.get("TOPN_TABLE", "sephora-lightfm-dev-topn")

_table = None


class _DynamoEncoder(json.JSONEncoder):
    """Serialize the ``Decimal`` values DynamoDB returns for every number."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return int(o) if o == o.to_integral_value() else float(o)
        return super().default(o)


def _get_table():
    """Create the DynamoDB resource once per container, not per request."""
    global _table
    if _table is None:
        if boto3 is None:
            raise RuntimeError("boto3 is unavailable; cannot reach DynamoDB.")
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        resource = (
            boto3.resource("dynamodb", endpoint_url=endpoint)
            if endpoint
            else boto3.resource("dynamodb")
        )
        _table = resource.Table(TOPN_TABLE)
    return _table


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, cls=_DynamoEncoder),
    }


def parse_request(event: dict[str, Any]) -> tuple[str, int]:
    """Extract and validate ``user_id`` and ``k`` from an API Gateway event.

    Split out from the handler so the validation rules are unit-testable
    without any AWS dependency.

    Args:
        event: API Gateway proxy event, or a bare dict for direct invocation.

    Returns:
        ``(user_id, k)``.

    Raises:
        ValueError: If ``user_id`` is missing or ``k`` is out of range.
    """
    params = event.get("queryStringParameters") or {}
    if not params and "user_id" in event:
        params = event

    user_id = params.get("user_id")
    if user_id in (None, ""):
        raise ValueError("Query parameter 'user_id' is required.")

    raw_k = params.get("k", DEFAULT_K)
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter 'k' must be an integer, got {raw_k!r}.") from None

    if not 1 <= k <= MAX_K:
        raise ValueError(f"Parameter 'k' must be between 1 and {MAX_K}, got {k}.")

    return str(user_id), k


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Return the stored top-N list for a user, truncated to ``k``."""
    try:
        user_id, k = parse_request(event)
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    try:
        item = _get_table().get_item(Key={"user_id": user_id}).get("Item")
    except Exception as exc:  # noqa: BLE001 - surface as 503, never leak a stack trace.
        return _response(503, {"error": "recommendation store unavailable", "detail": str(exc)})

    if not item:
        # Not in the precomputed table: either genuinely new, or waiting for
        # the next scheduled retrain to fold them in. The client is expected
        # to retry against the feature-based fallback path (see
        # recommender.serving.topn.score_new_user) using attributes supplied
        # at signup, not treat this as "no recommendations exist".
        return _response(
            404,
            {
                "error": f"no recommendations stored for user {user_id}",
                "user_id": user_id,
                "fallback": "score_from_profile_features",
            },
        )

    recommendations = item.get("recommendations", [])[:k]
    return _response(
        200,
        {
            "user_id": user_id,
            "k": k,
            "model": item.get("model", "hybrid_warp"),
            "recommendations": recommendations,
        },
    )
