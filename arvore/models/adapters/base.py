"""Adapter registry and helpers shared by model families."""

from collections.abc import Mapping
from contextlib import contextmanager

from carabiner.decorators import decorator_with_params

from ...typing import FitAdapter, Validation

ADAPTERS: dict[str, FitAdapter] = {}

PACKAGES: dict[str, str] = {}


def _class_path(obj):
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _backend_version(adapter_name: str) -> str:
    """Return the version of the installed codec backend."""
    from importlib import import_module

    module_name = PACKAGES[adapter_name]
    module = import_module(module_name)
    return module.__version__


@decorator_with_params
def register_adapter(
    fn: FitAdapter,
    name: str,
    ext: str | None = None,
    package: str | None = None
) -> FitAdapter:
    fn.name = name
    fn.filename = f"model.{ext or name}"
    ADAPTERS[name] = fn() if isinstance(fn, type) else fn
    PACKAGES[name] = package or name
    return fn


def _kwargs(fit_kwargs: Mapping[str, ...] | None):
    return dict(fit_kwargs or {})


def _require_validation(
    validation: Validation | None = None,
    rounds: int | None = None
):
    if rounds is not None:
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("early_stopping_rounds must be a positive integer")
        if validation is None:
            raise ValueError("Early stopping requires a training validation set")


def _require_native_loss(estimator, loss, censor=None):
    """Reject losses needing a custom objective, which this family lacks."""
    from ..losses import DEFAULT_LOSS

    if censor is not None:
        raise NotImplementedError(
            f"Censored targets are unsupported for {_class_path(estimator)}"
        )
    if loss is not None and loss.name != DEFAULT_LOSS:
        raise NotImplementedError(
            f"Custom loss {loss.name!r} is unsupported for {_class_path(estimator)}"
        )


def _drop_masked(loss, X, y, censor=None, fit_kwargs=None):
    """Drop rows with no finite target when the loss is masked.

    Row-aligned ``sample_weight`` in ``fit_kwargs`` is filtered in place;
    other row-aligned fit arguments are not.
    """
    if loss is None or not loss.masked:
        return X, y, censor

    import numpy as np

    observed = np.isfinite(y).reshape(len(y), -1).any(axis=1)
    if not observed.any():
        raise ValueError("No observed targets remain after masking")
    if fit_kwargs and fit_kwargs.get("sample_weight") is not None:
        fit_kwargs["sample_weight"] = np.asarray(fit_kwargs["sample_weight"])[observed]
    return (
        X[observed],
        y[observed],
        censor[observed] if censor is not None else None,
    )


@contextmanager
def _temporary_params(estimator, **params):
    """Set estimator parameters for the duration of a fit."""
    if not params:
        yield estimator
        return
    original = {
        key: estimator.get_params().get(key)
        for key in params
    }
    estimator.set_params(**params)
    try:
        yield estimator
    finally:
        estimator.set_params(**original)


def _normalize_leaf_indices(values, n_samples):
    """Normalise backend leaves to (samples, trees) int64."""
    import numpy as np

    leaves = np.asarray(values)

    if leaves.ndim == 0:
        raise ValueError("Expected an array of leaf indices")

    if leaves.ndim == 1:
        if n_samples == 1:
            leaves = leaves.reshape(1, -1)
        elif leaves.shape[0] == n_samples:
            leaves = leaves.reshape(n_samples, 1)
        else:
            raise ValueError("Leaf sample count mismatch")

    elif leaves.shape[0] == n_samples:
        leaves = leaves.reshape(n_samples, -1)

    else:
        raise ValueError("Leaf sample count mismatch")

    if leaves.shape[1] == 0:
        raise ValueError("Model has no trees")

    if not (
        np.issubdtype(leaves.dtype, np.integer)
        or np.issubdtype(leaves.dtype, np.floating)
    ):
        raise TypeError(
            f"Leaf indices must be numeric, got {leaves.dtype}"
        )

    if not np.all(np.isfinite(leaves)):
        raise ValueError("Non-finite leaf indices")

    if np.any(leaves < 0):
        raise ValueError("Negative leaf indices")

    if np.issubdtype(leaves.dtype, np.floating):
        if np.any(leaves != np.floor(leaves)):
            raise ValueError("Non-integral leaf indices")

    return leaves.astype(np.int64, copy=False)
