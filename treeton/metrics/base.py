from collections.abc import Callable

from carabiner.decorators import decorator_with_params
from numpy.typing import ArrayLike, NDArray

DEFAULT_METRICS: tuple[str] = (
    "rmsd",
    "mae",
)

METRICS_REGISTRY: dict[str, Callable] = {}


def _normalize_dims(
    x: ArrayLike, 
    y: ArrayLike,
    broadcast: bool = False,
    aggfun: Callable | None = None
) -> tuple[NDArray, NDArray]:
    """Normalize prediction dimensions.

    If `x` has one extra dimension compared to `y` and their leading
    dimensions match exactly, `x` is assumed to be an ensemble of
    predictions and the mean across its last axis is used. Otherwise the
    function ensures that `x` and `y` are broadcastable. and raises a
    ``ValueError`` if they are not.

    Parameters
    ==========
    x : ArrayLike
        Prediction array.
    y : ArrayLike
        Ground truth array.

    Returns
    =======
    Tuple[numpy.ndarray, numpy.ndarray]
        Possibly averaged `x` and original `y`.

    Raises
    ======
    ValueError
        If `x` and `y` are not broadcastable.
    
    """

    import numpy as np
    if aggfun is None:
        aggfun = np.mean

    x, y = np.asarray(x), np.asarray(y)

    if x.shape[:-1] != y.shape[:(x.ndim - 1)]:
        raise ValueError(
            f"Leading dimensions of x and y (shapes {x.shape} and {y.shape}) are not compatible!"
        )   

    if x.ndim == y.ndim + 1:
        x = aggfun(x, axis=-1)
    elif x.ndim == y.ndim and y.shape[-1] == 1:
        x = aggfun(x, axis=-1, keepdims=True)

    try:
        b = np.broadcast(x, y)
    except ValueError:
        raise ValueError(
            f"Shapes {x.shape} and {y.shape} are not broadcastable!"
        )

    if broadcast:
        bshape = b.shape
        x, y = (np.broadcast_to(v, bshape).ravel() for v in (x, y))

    return x, y


@decorator_with_params
def register_metric(fn, name):
    METRICS_REGISTRY[name] = fn
    return fn

@register_metric(name="rmsd")
def rmsd(y_pred, y_true):
    """Root mean square error/deviation.

    Parameters
    ==========
    x : numpy.ndarray
        Predicted values. If predictions are provided as an ensemble, the mean
        across the last axis is used.
    y : numpy.ndarray
        Ground truth values.

    Returns
    =======
    float
        The RMSE between ``x`` and ``y``.
    
    Examples
    ========
    >>> import numpy as np
    >>> pred = np.array([1.0, 2.0, 3.0])
    >>> truth = np.array([1.0, 2.0, 4.0])
    >>> np.allclose(rmse(pred, truth), np.sqrt(((pred - truth) ** 2).mean()))
    True

    """

    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


rmse = register_metric(rmsd, name="rmse")

@register_metric(name="mae")
def mae(y_pred, y_true):
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return np.mean(np.abs(y_pred - y_true))


@register_metric(name="pearson")
def pearson_r(y_pred, y_true):
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    return  np.corrcoef(y_pred, y_true)[0, 1]


@register_metric(name="spearman")
def spearman_r(y_pred, y_true):
    from scipy.stats import spearmanr
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    return spearmanr(
        y_pred, y_true,
        nan_policy="omit",
    ).statistic
