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
def rmsd(y_true, y_pred):
    """Root mean square error/deviation.

    Parameters
    ==========
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values. If predictions are provided as an ensemble, the mean
        across the last axis is used.

    Returns
    =======
    float
        The RMSE between ``y_true`` and ``y_pred``.
    
    Examples
    ========
    >>> import numpy as np
    >>> pred = np.array([1.0, 2.0, 3.0])
    >>> truth = np.array([1.0, 2.0, 4.0])
    >>> np.allclose(rmse(truth, pred), np.sqrt(((pred - truth) ** 2).mean()))
    True

    """
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


rmse = register_metric(rmsd, name="rmse")

@register_metric(name="mae")
def mae(y_true, y_pred):
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return np.mean(np.abs(y_pred - y_true))


@register_metric(name="pearson")
def pearson_r(y_true, y_pred):
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    return  np.corrcoef(y_pred, y_true)[0, 1]


@register_metric(name="spearman")
def spearman_r(y_true, y_pred):
    from scipy.stats import spearmanr
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    return spearmanr(
        y_pred, y_true,
        nan_policy="omit",
    ).statistic


# Additional regression metrics

@register_metric(name="mse")
def mse(y_true, y_pred):
    """Mean squared error.

    Examples
    ========
    >>> round(mse([1.0, 2.0, 4.0], [1.0, 2.0, 3.0]), 3)
    0.333

    """
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return float(np.mean(np.square(y_pred - y_true)))


@register_metric(name="medae")
def median_absolute_error(y_true, y_pred):
    """Median absolute error, robust to outlying residuals.

    Examples
    ========
    >>> median_absolute_error([1.0, 2.0, 10.0], [1.5, 2.0, 3.0])
    0.5

    """
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return float(np.median(np.abs(y_pred - y_true)))


@register_metric(name="max_error")
def max_error(y_true, y_pred):
    """Largest absolute residual.

    Examples
    ========
    >>> max_error([1.0, 2.0, 10.0], [1.5, 2.0, 3.0])
    7.0

    """
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return float(np.max(np.abs(y_pred - y_true)))


@register_metric(name="bias")
def mean_bias(y_true, y_pred):
    """Mean signed error, ``mean(y_pred - y_true)``.

    Positive values mean the model overpredicts on average.

    Examples
    ========
    >>> mean_bias([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    1.0

    """
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return float(np.mean(y_pred - y_true))


@register_metric(name="r2")
def r2(y_true, y_pred):
    """Coefficient of determination, ``1 - SS_res / SS_tot``.

    Can be negative when the model is worse than predicting the mean.
    Multi-target arrays are pooled, as for ``pearson``.

    Examples
    ========
    >>> r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0
    >>> r2([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    0.0

    """
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    ss_tot = np.sum(np.square(y_true - y_true.mean()))
    if ss_tot == 0:
        return float("nan")
    return float(1. - np.sum(np.square(y_true - y_pred)) / ss_tot)


@register_metric(name="explained_variance")
def explained_variance(y_true, y_pred):
    """Explained variance, ``1 - Var(y_true - y_pred) / Var(y_true)``.

    Unlike ``r2``, it ignores a constant offset between predictions
    and targets; compare the two with ``bias``.

    Examples
    ========
    >>> explained_variance([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    1.0

    """
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    var = np.var(y_true)
    if var == 0:
        return float("nan")
    return float(1. - np.var(y_true - y_pred) / var)


@register_metric(name="ccc")
def concordance_correlation(y_true, y_pred):
    """Lin's concordance correlation coefficient.

    Agreement with the identity line: penalises scale and location
    shifts that Pearson's r ignores.

    Examples
    ========
    >>> concordance_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0
    >>> round(concordance_correlation([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]), 3)
    0.571

    """
    import numpy as np
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denominator = y_true.var() + y_pred.var() + (mean_t - mean_p) ** 2
    if denominator == 0:
        return float("nan")
    return float(2. * cov / denominator)


@register_metric(name="kendall")
def kendall_tau(y_true, y_pred):
    """Kendall's tau-b rank correlation.

    Examples
    ========
    >>> kendall_tau([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    1.0

    """
    from scipy.stats import kendalltau
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    return float(kendalltau(y_true, y_pred, nan_policy="omit").statistic)


# Classification metrics on regression outputs.
#
# Labels are binarized at a cutoff on y_true: a fixed value
# (``threshold``) or the top ``quantile`` fraction of y_true.
# Ranking metrics (AUROC, average precision, enrichment) use the
# continuous predictions as scores. Confusion-matrix metrics also
# binarize predictions at the same cutoff.

def _cutoff(y_true, *, threshold=None, quantile=None):
    import numpy as np

    if (threshold is None) == (quantile is None):
        raise ValueError("Give exactly one of threshold or quantile")
    if threshold is not None:
        return float(threshold)
    if not 0. < quantile < 1.:
        raise ValueError(f"quantile must be in (0, 1): {quantile}")
    return float(np.quantile(y_true, 1. - quantile))


def _binary(y_true, y_pred, *, threshold, quantile, lower):
    """Return (labels, scores, cutoff), with positives as the high class."""
    y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
    cutoff = _cutoff(y_true, threshold=threshold, quantile=quantile)
    if lower:
        # Flip so that "lower is positive" becomes "higher is positive".
        y_true, y_pred, cutoff = -y_true, -y_pred, -cutoff
    return y_true >= cutoff, y_pred, cutoff


def _confusion(labels, predicted):
    import numpy as np
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    tn = int(np.sum(~labels & ~predicted))
    return tp, fp, fn, tn


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def _roc_auc(labels, scores):
    from scipy.stats import rankdata
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)  # average ranks handle ties
    return float(
        (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def _average_precision(labels, scores):
    """Step-wise average precision; tied scores form one threshold."""
    import numpy as np
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    scores, hits = scores[order], labels[order]
    # Last index of each run of tied scores.
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(hits)[ends]
    precision = tp / (ends + 1)
    recall_gain = np.diff(np.r_[0, tp]) / n_pos
    return float(np.sum(recall_gain * precision))


def _enrichment(labels, scores, *, fraction):
    """Hit rate in the top-scored fraction over the overall hit rate."""
    import numpy as np
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    k = max(1, int(round(fraction * len(labels))))
    top = np.argsort(-scores, kind="stable")[:k]
    return float((labels[top].sum() / k) / (n_pos / len(labels)))


def binarized_metric(
    kind: str,
    *,
    threshold: float | None = None,
    quantile: float | None = None,
    lower: bool = False,
    fraction: float | None = None,
) -> Callable:
    """Build a classification metric for continuous targets.

    Parameters
    ==========
    kind : str
        One of ``"roc_auc"``, ``"average_precision"``, ``"enrichment"``,
        ``"precision"``, ``"recall"``, ``"f1"``, ``"mcc"``,
        ``"balanced_accuracy"``.
    threshold : float, optional
        Fixed cutoff on ``y_true`` defining positives.
    quantile : float, optional
        Fraction of ``y_true`` (top, or bottom when ``lower=False``)
        defining positives. Give exactly one of threshold or quantile.
    lower : bool
        Whether values at or beyond the cutoff in the high direction are
        positive. Use ``False`` when low values are hits (e.g. MIC).
    fraction : float, optional
        Top-scored fraction for ``"enrichment"``. Defaults to the
        positive fraction implied by ``quantile``.

    Returns
    =======
    Callable
        ``metric(y_true, y_pred) -> float``.

    Examples
    ========
    >>> import numpy as np
    >>> y_true = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    >>> y_pred = np.array([0.2, 0.3, 0.45, 0.7, 0.95])
    >>> auroc = binarized_metric("roc_auc", threshold=0.5)
    >>> auroc(y_true, y_pred)
    1.0
    >>> f1 = binarized_metric("f1", threshold=0.5)
    >>> f1(y_true, y_pred)
    1.0
    >>> low_is_hit = binarized_metric("roc_auc", threshold=0.5, lower=False)
    >>> low_is_hit(y_true, y_pred)
    1.0

    """
    import numpy as np

    ranking = {"roc_auc", "average_precision", "enrichment"}
    confusion = {
        "precision", "recall", "f1", "mcc", "balanced_accuracy",
    }
    if kind not in ranking | confusion:
        raise ValueError(f"Unknown binarized metric: {kind}")
    _cutoff([0., 1.], threshold=threshold, quantile=quantile)  # validate
    if fraction is not None and not 0. < fraction <= 1.:
        raise ValueError(f"fraction must be in (0, 1]: {fraction}")

    def metric(y_true, y_pred):
        labels, scores, cutoff = _binary(
            y_true, y_pred,
            threshold=threshold, quantile=quantile, lower=lower,
        )
        if kind == "roc_auc":
            return _roc_auc(labels, scores)
        if kind == "average_precision":
            return _average_precision(labels, scores)
        if kind == "enrichment":
            top = fraction if fraction is not None else (
                quantile if quantile is not None else labels.mean()
            )
            return _enrichment(labels, scores, fraction=top)

        tp, fp, fn, tn = _confusion(labels, scores >= cutoff)
        if kind == "precision":
            return _divide(tp, tp + fp)
        if kind == "recall":
            return _divide(tp, tp + fn)
        if kind == "f1":
            return _divide(2 * tp, 2 * tp + fp + fn)
        if kind == "balanced_accuracy":
            return float(np.nanmean([
                _divide(tp, tp + fn), _divide(tn, tn + fp),
            ]))
        # mcc
        denominator = np.sqrt(
            float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        return _divide(tp * tn - fp * fn, denominator)

    where = (
        f"threshold={threshold}" if threshold is not None
        else f"quantile={quantile}"
    )
    metric.__name__ = f"{kind}[{where},lower={lower}]"
    metric.__doc__ = f"{kind} with positives at {where} (lower={lower})."
    return metric


# Top-10% of y_true as positives: a common "hit" definition for screens.
for _kind in (
    "roc_auc", "average_precision", "enrichment",
    "precision", "recall", "f1", "mcc", "balanced_accuracy",
):
    register_metric(
        binarized_metric(_kind, quantile=.1),
        name=f"{_kind}_top10",
    )
del _kind
