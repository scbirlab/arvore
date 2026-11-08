from collections.abc import Callable
from carabiner.decorators import decorator_with_params

DEFAULT_METRICS: tuple[str] = (
    "rmsd",
    "mae",
)

METRICS_REGISTRY: dict[str, Callable] = {}

@decorator_with_params
def register_metric(fn, name):
    METRICS_REGISTRY[name] = fn
    return fn

@register_metric("rmsd")
def rmsd(y_pred, y_true):
    import numpy as np
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

@register_metric("mae")
def mae(y_pred, y_true):
    import numpy as np
    return np.mean(np.abs(y_pred - y_true))
