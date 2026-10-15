"""Declarative feature specification loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_FAMILIES = {"user", "item", "interaction", "context"}
VALID_KINDS = {"categorical", "multi_categorical", "identity"}
VALID_FIT_SCOPES = {"catalogue", "profile", "train_only"}


@dataclass(frozen=True)
class FeatureSpec:
    """A single declared feature -- a rule for producing vocabulary tokens."""

    name: str
    family: str
    kind: str
    module: str
    transform: str
    params: dict[str, Any]
    fit_on: str
    point_in_time: bool
    fallback: str | None
    rationale: str

    def __post_init__(self) -> None:
        if self.family not in VALID_FAMILIES:
            raise ValueError(
                f"Feature '{self.name}' declares unknown family '{self.family}'. "
                f"Valid families: {sorted(VALID_FAMILIES)}"
            )
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"Feature '{self.name}' declares unknown kind '{self.kind}'. "
                f"Valid kinds: {sorted(VALID_KINDS)}"
            )
        if self.fit_on not in VALID_FIT_SCOPES:
            raise ValueError(
                f"Feature '{self.name}' declares unknown fit scope '{self.fit_on}'. "
                f"Valid scopes: {sorted(VALID_FIT_SCOPES)}"
            )
        if self.kind == "identity" and self.fit_on != "train_only":
            raise ValueError(
                f"Feature '{self.name}' is an identity feature but fit_on is "
                f"'{self.fit_on}', not 'train_only'. Identity vocabularies must be "
                "restricted to the training partition, otherwise cold entities "
                "would keep an ID token and cold-start would not be a real test."
            )
        if not self.rationale.strip():
            raise ValueError(f"Feature '{self.name}' must document a rationale.")


def default_spec_path() -> Path:
    """Return the repository's canonical feature specification path."""
    # spec.py -> features -> recommender -> src -> repository root
    return Path(__file__).resolve().parents[3] / "conf" / "features.yaml"


def load_feature_specs(path: str | Path | None = None) -> list[FeatureSpec]:
    """Parse and validate ``conf/features.yaml``.

    Args:
        path: Optional override; defaults to the repository specification.

    Returns:
        Validated specifications in declaration order.

    Raises:
        FileNotFoundError: If the specification file is missing.
        ValueError: If the document is malformed or a name is duplicated.
    """
    path = Path(path) if path is not None else default_spec_path()
    if not path.exists():
        raise FileNotFoundError(f"Feature specification not found at {path}")

    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or "features" not in document:
        raise ValueError(f"{path} must contain a top-level 'features' list.")

    specs: list[FeatureSpec] = []
    seen: set[str] = set()
    for entry in document["features"]:
        name = entry.get("name")
        if not name:
            raise ValueError(f"{path} contains a feature without a name.")
        if name in seen:
            raise ValueError(f"Duplicate feature declaration: '{name}'")
        seen.add(name)

        specs.append(
            FeatureSpec(
                name=name,
                family=entry["family"],
                kind=entry["kind"],
                module=entry["module"],
                transform=entry["transform"],
                params=entry.get("params") or {},
                fit_on=entry.get("fit_on", "catalogue"),
                point_in_time=bool(entry.get("point_in_time", False)),
                fallback=entry.get("fallback"),
                rationale=entry.get("rationale", ""),
            )
        )

    if not specs:
        raise ValueError(f"{path} declares no features.")
    return specs
