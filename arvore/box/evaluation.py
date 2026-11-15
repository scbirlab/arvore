"""Targets, censoring indicators and metrics for Box evaluation."""

from collections.abc import Callable, Iterable, Mapping

import numpy as np

from ..metrics import DEFAULT_METRICS, METRICS_REGISTRY


def resolve_metrics(
    metrics: str | Callable | Iterable[str | Callable] | Mapping[str, str | Callable] | None = None,
) -> dict[str, Callable]:
    """Resolve metric names and callables to ``{name: callable}``."""
    if metrics is None:
        metrics = DEFAULT_METRICS
    if isinstance(metrics, str) or callable(metrics):
        metrics = [metrics]
    if not isinstance(metrics, Mapping):
        metrics = {
            metric if isinstance(metric, str) else getattr(metric, "__name__", type(metric).__name__): metric
            for metric in metrics
        }

    resolved = {}
    for name, metric in metrics.items():
        if isinstance(metric, str):
            try:
                metric = METRICS_REGISTRY[metric]
            except KeyError as error:
                raise ValueError(
                    f"Unknown metric {metric!r}. Available: {sorted(METRICS_REGISTRY)}"
                ) from error
        elif not callable(metric):
            raise TypeError(
                f"Metrics must be callable or registered names, but was {type(metric)}: {metric}"
            )
        resolved[name] = metric
    return resolved


def targets_and_censor(arrays):
    """Return target and optional censoring indicators, 1D for a single target."""
    # sklearn regressors generally expect a 1D single target.
    y, *censor = (array[:, 0] if array.shape[1] == 1 else array for array in arrays)
    return y, (censor[0] if censor else None)


def exact_observed(y_true, y_pred, censor=None, masked=False):
    """Select exact targets, and observed ones when masked, for scoring.

    Several targets are flattened to their selected values.
    """
    selected = np.ones(np.shape(y_true), dtype=bool)
    if masked:
        selected &= np.isfinite(y_true)
    if censor is not None:
        selected &= ~((censor < 0) | (censor > 0))
    if selected.all():
        return y_true, y_pred
    if not selected.any():
        raise ValueError("No exact, observed targets to evaluate")
    return y_true[selected], y_pred[selected]
