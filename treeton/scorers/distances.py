"""Nearest-reference distances in model feature space."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import ScoreContext
from .registry import register_scorer


@dataclass(frozen=True)
class FeatureReference:
    """Exact nearest-neighbour index over transformed training features."""

    index: Any
    features: NDArray
    metric: str

    @property
    def n_reference(self):
        return len(self.features)

    @property
    def n_features(self):
        return self.features.shape[1]

    @classmethod
    def build(cls, X, *, metric="euclidean"):
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or not all(X.shape):
            raise ValueError("Reference features must be nonempty 2D")
        if not np.all(np.isfinite(X)):
            raise ValueError("Reference features must be finite")
        if not isinstance(metric, str) or not metric:
            raise ValueError("metric must be a nonempty sklearn metric name")

        index = NearestNeighbors(
            metric=metric,
            algorithm="auto",
        ).fit(X)

        return cls(index=index, features=X, metric=metric)

    def state_dict(self):
        return {"features": np.asarray(self.features, dtype=np.float64)}

    @classmethod
    def from_state_dict(cls, state, spec):
        features = np.asarray(state["features"])
        if features.dtype != np.float64:
            raise ValueError("Expected float64 feature reference")
        return cls.build(features, metric=spec["metric"])

    def distance(self, X, *, n_neighbors=1, return_var=False):
        """Mean distance to k nearest reference rows."""
        if not isinstance(n_neighbors, int) or not (
            1 <= n_neighbors <= self.n_reference
        ):
            raise ValueError("Invalid number of neighbours")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected (samples, {self.n_features}), got {X.shape}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("Candidate features must be finite")
        if len(X) == 0:
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty.copy()) if return_var else empty

        distances, _ = self.index.kneighbors(
            X, n_neighbors=n_neighbors
        )
        mean = distances.mean(axis=1)

        if return_var:
            return mean, distances.var(axis=1, ddof=0)
        return mean


@register_scorer
@dataclass(frozen=True)
class FeatureDistance:
    """Mean k-neighbour distance in transformed feature space."""

    name: ClassVar[str] = "feature_distance"
    version: ClassVar[int] = 1

    metric: str = "euclidean"
    n_neighbors: int = 1
    return_var: bool = False

    def __post_init__(self):
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("metric must be a nonempty string")
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")

    def prepare(self, box, *, batch_size=1024):
        from .reference import feature_batches

        def build():
            blocks = [
                np.asarray(X, dtype=np.float64)
                for X in feature_batches(box, batch_size=batch_size)
            ]
            if not blocks:
                raise ValueError("Training reference dataset is empty")
            return FeatureReference.build(
                np.concatenate(blocks, axis=0),
                metric=self.metric,
            )

        return box.references.get_or_build(
            {
                "type": "feature_distance",
                "version": self.version,
                "features": list(box.features),
                "metric": self.metric,
                "algorithm": "auto",
                "dtype": "float64",
            },
            build,
        )

    def __call__(self, context: ScoreContext) -> NDArray:
        if context.state is None:
            raise ValueError("Feature reference has not been prepared")
        result = context.state.distance(
            context.X,
            n_neighbors=self.n_neighbors,
            return_var=self.return_var,
        )
        if self.return_var:
            mean, variance = result
            return {"": mean, "var": variance}
        return result
