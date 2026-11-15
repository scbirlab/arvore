"""Resolve adapters from estimators and saved codec names."""

from ...typing import Estimator, FitAdapter
from .base import ADAPTERS, _class_path


def get_adapter(estimator: Estimator) -> FitAdapter:
    """Select the registered family handling ``estimator``, else scikit-learn.

    Raises ``TypeError`` when more than one registered family handles it.
    """
    matches = [
        adapter for adapter in ADAPTERS.values()
        if adapter.handles(estimator)
    ]
    if len(matches) > 1:
        raise TypeError(
            f"Several adapters handle {_class_path(estimator)}: "
            f"{[adapter.name for adapter in matches]}"
        )
    return matches[0] if matches else ADAPTERS["skops"]


def get_adapter_by_name(name: str):
    """Resolve a saved backend identifier."""
    try:
        return ADAPTERS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model codec: {name}"
        ) from error


def get_serialization_adapter(model, *, allow_unsafe=False):
    """Resolve persistence through the estimator's backend."""
    adapter = get_adapter(model)
    name = adapter.adapter_name(
        model,
        allow_unsafe=allow_unsafe,
    )
    return get_adapter_by_name(name)
