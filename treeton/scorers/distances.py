"""Nearest-reference distance diagnostics."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import ScoreContext
from .registry import register_scorer



@dataclass(frozen=True)
class EuclideanReference:
    """Fitted exact nearest-neighbour reference."""

    index: Any
    n_reference: int
    n_features: int
    features: NDArray

    @classmethod
    def build(cls, X) -> "EuclideanReference":
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or not all(X.shape):
            raise ValueError("Reference features must be nonempty 2D")

        if not np.all(np.isfinite(X)):
            raise ValueError("Reference features must be finite")

        index = NearestNeighbors(
            metric="euclidean",
            algorithm="auto",
        ).fit(X)

        return cls(
            index=index,
            n_reference=len(X),
            n_features=X.shape[1],
            features=X,
        )

    def state_dict(self):
        """Persist features; reconstruct NN index on load."""
        return {
            "features": np.asarray(
                self.features,
                dtype=np.float64,
            ),
        }

    @classmethod
    def from_state_dict(cls, state, spec):
        features = np.asarray(state["features"])
        if features.dtype != np.float64:
            raise ValueError("Expected float64 Euclidean features")
        return cls.build(features)

    def distance(self, X, *, n_neighbors: int = 1) -> NDArray:
        """Mean distance to the k nearest training observations."""
        X = np.asarray(X, dtype=np.float64)

        if not 1 <= n_neighbors <= self.n_reference:
            raise ValueError("Invalid number of neighbours")

        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected (samples, {self.n_features}), got {X.shape}"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("Candidate features must be finite")

        if not len(X):
            return np.empty(0, dtype=float)

        distances, _ = self.index.kneighbors(
            X,
            n_neighbors=n_neighbors,
        )
        return distances.mean(axis=1)



@register_scorer
@dataclass(frozen=True)
class EuclideanDistance:
    """Distance to the nearest training reference in feature space."""

    n_neighbors: int = 1
    name: ClassVar[str] = "euclidean_distance"
    version: ClassVar[str] = 1


    def __post_init__(self):
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")


    def prepare(self, box, *, batch_size=1024):
        """Build or reuse the training-reference nearest-neighbour index."""
        from .reference import feature_batches

        def build():
            blocks = [
                np.asarray(X, dtype=np.float64)
                for X in feature_batches(box, batch_size=batch_size)
            ]
            if not blocks:
                raise ValueError("Training reference dataset is empty")
            return EuclideanReference.build(
                np.concatenate(blocks, axis=0)
            )

        return box.references.get_or_build(
            {
                "type": "euclidean",
                "version": self.version,
                "features": list(box.features),
                "metric": "euclidean",
                "algorithm": "auto",
                "dtype": "float64",
            }, 
            build,
        )

    def __call__(self, context: ScoreContext) -> NDArray:
        reference = context.state

        if reference is None:
            raise ValueError(
                "euclidean_distance requires a fitted "
                "training-reference index"
            )

        return reference.distance(context.X)
