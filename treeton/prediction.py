"""Prediction execution shared by Box and optional scorers."""

from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .models.adapters import get_adapter
from .models.ensemble import Ensemble


@dataclass(frozen=True)
class PredictionBatch:
    predictions: NDArray
    columns: Mapping[str, NDArray]
    member_predictions: NDArray | None = None
    member_weights: NDArray | None = None
    tree_predictions: tuple[NDArray, ...] | None = None
    resources: Mapping = field(default_factory=dict)


def predict_batch(
    model,
    X,
    *,
    targets,
    prediction_column="prediction",
    kwargs=None,
    needs_trees=False,
    needs_leaves=False,
):
    """Calculate predictions and requested shared model resources."""
    kwargs = dict(kwargs or {})
    disaggregate = kwargs.pop("disaggregate", False)

    is_ensemble = isinstance(model, Ensemble)
    members = (
        tuple(model.members.values())
        if is_ensemble
        else (model,)
    )

    tree_predictions = None
    resources = {}

    if needs_trees:
        tree_predictions = tuple(
            np.asarray(
                get_adapter(member).tree_predictions(member, X),
                dtype=float,
            )
            for member in members
        )

    if needs_leaves:
        resources["leaf_indices"] = tuple(
            np.asarray(
                get_adapter(member).leaf_indices(member, X)
            )
            for member in members
        )

    member_predictions = None
    member_weights = None

    if is_ensemble:
        member_weights = np.asarray(
            model.weights, dtype=float
        )

        if tree_predictions is not None and not kwargs:
            member_predictions = np.stack(
                [
                    values.mean(axis=1)
                    for values in tree_predictions
                ],
                axis=1,
            )
        else:
            member_predictions = model.predict_members(
                X, **kwargs
            )

        predictions = np.average(
            member_predictions,
            axis=1,
            weights=member_weights,
        )

    elif tree_predictions is not None and not kwargs:
        predictions = tree_predictions[0].mean(axis=1)

    else:
        predictions = get_adapter(model).predict(
            model, X, **kwargs
        )

    predictions = np.asarray(predictions)

    if predictions.ndim == 1:
        predictions = predictions[:, None]

    expected = (len(X), len(targets))

    if predictions.shape != expected:
        raise ValueError(
            f"Expected predictions with shape {expected}, "
            f"received {predictions.shape}"
        )

    if disaggregate and is_ensemble:
        member_shape = (
            len(X),
            len(model.members),
            len(targets),
        )
        if member_predictions.shape != member_shape:
            raise ValueError(
                "Invalid ensemble member prediction shape: "
                f"{member_predictions.shape}, expected {member_shape}"
            )

        columns = {
            f"{prediction_column}:{target}:{name}":
                member_predictions[:, member_idx, target_idx]
            for target_idx, target in enumerate(targets)
            for member_idx, name in enumerate(model.members)
        }

    else:
        columns = {
            f"{prediction_column}:{target}": values
            for target, values in zip(
                targets, predictions.T
            )
        }

    return PredictionBatch(
        predictions=predictions,
        columns=columns,
        member_predictions=member_predictions,
        member_weights=member_weights,
        tree_predictions=tree_predictions,
        resources=resources,
    )
