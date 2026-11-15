"""Prediction execution shared by Box and optional scorers."""

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from .models.adapters import get_adapter
from .models.ensemble import Ensemble
from .tabular import matrix

DEFAULT_PREDICTION_COLUMN: str = "prediction"


@dataclass(frozen=True)
class PredictionBatch:
    predictions: NDArray
    columns: Mapping[str, NDArray]
    member_predictions: NDArray | None = None
    member_weights: NDArray | None = None
    tree_predictions: tuple[NDArray, ...] | None = None
    resources: Mapping = field(default_factory=dict)


def predict_batch(
    model: Ensemble,
    X,
    *,
    targets,
    prediction_column=DEFAULT_PREDICTION_COLUMN,
    kwargs=None,
    needs_trees=False,
    needs_leaves=False,
):
    """Calculate ensemble predictions and requested shared model resources."""
    kwargs = dict(kwargs or {})
    disaggregate = kwargs.pop("disaggregate", False)
    members = tuple(model.members.values())

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

    member_weights = np.asarray(model.weights, dtype=float)

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

    expected = (len(X), len(targets))

    if predictions.shape != expected:
        raise ValueError(
            f"Expected predictions with shape {expected}, "
            f"received {predictions.shape}"
        )

    if disaggregate and len(members) > 1:
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


def predict_map_batch(
    batch: Mapping[str, ...],
    *,
    model,
    column_groups: Iterable[Iterable[str]],
    pipeline,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    kwargs: Mapping | None = None,
    scoring=None,
) -> dict[str, ...]:
    """Predict, and optionally score, one batch of a Dataset map."""
    from .scorers import ScoreContext

    feature_cols, target_cols = column_groups

    (X,) = matrix(
        batch,
        pipeline,
        column_groups=(feature_cols,),
    )

    result = predict_batch(
        model,
        X,
        targets=tuple(target_cols),
        prediction_column=prediction_column,
        kwargs=kwargs,
        needs_trees=(
            scoring.needs_trees if scoring else False
        ),
        needs_leaves=(
            scoring.needs_leaves if scoring else False
        ),
    )

    output = batch | dict(result.columns)

    if scoring is None:
        return output

    context = ScoreContext(
        box=scoring.box,
        X=X,
        predictions=result.predictions,
        model=model,
        batch=batch,
        targets=tuple(target_cols),
        member_predictions=result.member_predictions,
        member_weights=result.member_weights,
        tree_predictions=result.tree_predictions,
        resources=result.resources,
    )

    output.update(
        scoring.score(
            context,
            occupied=output.keys(),
        )
    )

    return output
