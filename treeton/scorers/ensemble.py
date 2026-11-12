"""Diagnostics based on ensemble member predictions."""

from typing import ClassVar
from dataclasses import dataclass

from numpy.typing import NDArray

from .base import Requirements, ScoreContext
from .registry import register_scorer


@register_scorer
@dataclass(frozen=True)
class EnsembleSD:
    """Weighted population SD across ensemble members."""

    name: ClassVar[str] = "ensemble_sd"

    @property
    def requirements(self) -> Requirements:
        return Requirements(
            outputs=("member_predictions",),
            capabilities=("ensemble_members",),
        )

    def __call__(
        self, 
        context: ScoreContext
    ) -> NDArray:
        import numpy as np

        predictions = context.member_predictions

        if predictions is None:
            raise ValueError(
                "ensemble_sd requires member predictions"
            )

        weights = context.member_weights
        if weights is None:
            weights = np.ones(
                predictions.shape[1],
                dtype=float,
            )
        
        predictions = np.asarray(predictions, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if predictions.ndim != 3:
            raise ValueError(
                "Expected member predictions with shape "
                "`(samples, members, targets)`"
            )

        n_samples, n_members, n_targets = predictions.shape

        if n_members < 2:
            raise ValueError(
                "`ensemble_sd` requires at least two members"
            )

        if weights.shape != (n_members,):
            raise ValueError(
                f"Expected {n_members} member weights, "
                f"received shape {weights.shape}"
            )

        if (
            not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError(
                f"Member weights must be positive and finite, but were {weights}"
            )

        if n_samples != len(context.X):
            raise ValueError(
                f"Sample count mismatch: {n_samples=}, {len(context.X)=}"
            )

        if n_targets != len(context.targets):
            raise ValueError(
                f"Target count mismatch: {n_targets=}, {len(context.targets)}"
            )

        mean = np.average(
            predictions,
            axis=1,
            weights=weights,
        )
        variance = np.average(
            (predictions - mean[:, None, :]) ** 2.,
            axis=1,
            weights=weights,
        )
        return np.sqrt(variance)
