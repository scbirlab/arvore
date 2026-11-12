"""Nearest-reference distance diagnostics."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import Requirements, ScoreContext
from .registry import register_scorer


@dataclass(frozen=True)
class EuclideanReference:
    """Fitted exact nearest-neighbour index."""

    index: Any
    n_reference: int
    n_features: int

    @classmethod
    def build(cls, X) -> "EuclideanReference":
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError(
                "Reference features must be a nonempty 2D array"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Reference features must be finite"
            )

        index = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
            algorithm="auto",
        )
        index.fit(X)

        return cls(
            index=index,
            n_reference=len(X),
            n_features=X.shape[1],
        )

    def distance(self, X) -> NDArray:
        """Return nearest-reference distance per candidate."""
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected candidate features with shape "
                f"(samples, {self.n_features}); got {X.shape}"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Candidate features must be finite"
            )

        if not len(X):
            return np.empty(0, dtype=float)

        distances, _ = self.index.kneighbors(
            X,
            n_neighbors=1,
            return_distance=True,
        )

        return distances[:, 0]


@register_scorer
@dataclass(frozen=True)
class EuclideanDistance:
    """Distance to the nearest training reference in feature space."""

    name: ClassVar[str] = "euclidean_distance"

    @property
    def requirements(self) -> Requirements:
        return Requirements(
            reference_features=True,
            reference_indices=("euclidean",),
        )

    def prepare(self, box, *, batch_size=1024):
        """Build the exact training-reference NN index."""
        from .reference import feature_batches

        blocks = [
            np.asarray(X, dtype=np.float64)
            for X in feature_batches(
                box, batch_size=batch_size
            )
        ]

        if not blocks:
            raise ValueError(
                "Training reference dataset is empty"
            )

        return EuclideanReference.build(
            np.concatenate(blocks, axis=0)
        )

    def __call__(self, context: ScoreContext) -> NDArray:
        reference = context.state

        if reference is None:
            raise ValueError(
                "euclidean_distance requires a fitted "
                "training-reference index"
            )

        return reference.distance(context.X)
