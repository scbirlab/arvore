"""Diagnostics based on ensemble member predictions."""

from typing import ClassVar
from dataclasses import dataclass
import warnings

import numpy as np
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

    def __call__(self, context: ScoreContext) -> NDArray:
        predictions = context.member_predictions

        if predictions.shape[1] < 2:
            raise ValueError(
                "`ensemble_sd` requires at least two members"
            )

        mean = np.average(
            predictions,
            axis=1,
            weights=context.member_weights,
        )
        variance = np.average(
            (predictions - mean[:, None, :]) ** 2.,
            axis=1,
            weights=context.member_weights,
        )
        return np.sqrt(variance)

    def finalize(self, result, *, name, box):
        """Warn when all ensemble-disagreement scores are zero."""
        columns = [
            f"score:{name}:{target}"
            for target in box.target
        ]

        if len(result) == 0:
            return

        missing = set(columns) - set(result.column_names)
        if missing:
            raise ValueError(
                f"Missing ensemble SD columns: {sorted(missing)}"
            )

        for batch in result.select_columns(columns).iter(
            batch_size=4096
        ):
            if any(
                np.any(np.asarray(batch[column]) != 0)
                for column in columns
            ):
                return

        warnings.warn(
            f"Scorer {name!r}: ensemble disagreement is zero "
            "for every candidate. Members may be identical. "
            "For repeated XGBoost, consider subsample < 1 or "
            "colsample_bytree < 1 with different random seeds.",
            RuntimeWarning,
            stacklevel=2,
        )


@register_scorer
@dataclass(frozen=True)
class TreeSD:
    """Root mean within-forest variance across ensemble members."""

    name: ClassVar[str] = "tree_sd"

    @property
    def requirements(self) -> Requirements:
        return Requirements(
            outputs=("tree_predictions",),
            capabilities=("tree_predictions",),
        )

    def __call__(self, context: ScoreContext) -> NDArray:
        forests = context.tree_predictions

        if any(predictions.shape[1] < 2 for predictions in forests):
            raise ValueError(
                "tree_sd requires at least two trees per forest"
            )

        variances = np.stack(
            [np.var(predictions, axis=1, ddof=0) for predictions in forests],
            axis=1,
        )
        return np.sqrt(
            np.average(variances, axis=1, weights=context.member_weights)
        )
