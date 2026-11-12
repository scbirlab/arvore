"""Interfaces for pointwise prediction diagnostics."""

from typing import Any, Protocol
from collections.abc import Mapping
from dataclasses import dataclass, field

from numpy.typing import NDArray


@dataclass(frozen=True)
class Requirements:
    """Resources required to evaluate a scorer."""

    columns: tuple[str, ...] = field(default_factory=tuple)
    reference_columns: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = ("prediction",)
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    reference_features: bool = False
    reference_indices: tuple[str, ...] = field(default_factory=tuple)
    reference_score_distribution: bool = False


@dataclass(frozen=True)
class ScoreContext:
    """Backend-independent inputs supplied to a scorer."""

    X: NDArray
    predictions: NDArray
    model: Any
    batch: Mapping[str, ...]
    targets: tuple[str, ...]

    member_predictions: NDArray | None = None
    member_weights: NDArray | None = None
    tree_predictions: tuple[NDArray, ...] | None = None
    resources: Mapping[str, Any] = field(default_factory=dict)
    box: Any = None
    state: Any = None


class Scorer(Protocol):
    """Callable producing one score per sample or target."""

    name: str

    @property
    def requirements(self) -> Requirements:
        ...

    def __call__(self, context: ScoreContext) -> NDArray:
        ...
