"""Censored and masked regression losses."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import cached_property

from carabiner.decorators import decorator_with_params
from numpy.typing import ArrayLike

DEFAULT_LOSS: str = "squared_error"

LOSSES: dict[str, Callable] = {}


@decorator_with_params
def register_loss(fn, name):
    LOSSES[name] = fn
    return fn


@register_loss(name=DEFAULT_LOSS)
def squared_error(prediction, lower, upper):
    """Half the squared distance from each prediction to its target interval.

    Exact targets have equal bounds, censored targets one infinite bound
    and masked targets two, so predictions on the censored side of a
    bound, and all predictions of masked targets, incur no loss.

    Returns
    =======
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Elementwise loss, and its gradient and Hessian with respect
        to ``prediction``.

    Examples
    ========
    >>> import numpy as np
    >>> loss, gradient, hessian = squared_error(
    ...     np.array([1., 1., 1.]),
    ...     np.array([3., 0., -np.inf]),
    ...     np.array([3., np.inf, 0.]),
    ... )
    >>> loss.tolist(), gradient.tolist(), hessian.tolist()
    ([2.0, 0.0, 0.5], [-2.0, 0.0, 1.0], [1.0, 0.0, 1.0])

    """
    import numpy as np

    residual = prediction - np.clip(prediction, lower, upper)
    hessian = ((residual != 0.) | (lower == upper)).astype(float)
    return np.square(residual) / 2., residual, hessian


def _row_weights(weight, shape):
    import numpy as np

    if weight is None:
        return np.ones(shape)
    return np.broadcast_to(
        np.reshape(weight, (-1,) + (1,) * (len(shape) - 1)),
        shape,
    )


@dataclass(frozen=True)
class Loss:
    """Training loss for a Box.

    With the default ``name``, every backend trains on its estimator's
    own objective where that suffices: rows missing every target are
    dropped (with any ``sample_weight``; other row-aligned fit arguments
    are not filtered). Censored targets, targets partially missing
    across several targets, and any other ``name`` need a custom
    objective, which XGBoost and LightGBM train on and other backends
    reject with ``NotImplementedError``.

    Parameters
    ==========
    name : str, optional
        Registered loss, or import path of a function
        ``f(prediction, lower, upper) -> (loss, gradient, hessian)``
        over target intervals. Default: ``"squared_error"``.
    masked : bool, optional
        Ignore missing targets. Default: ``False``.

    Examples
    ========
    >>> Loss(masked=True).to_dict()
    {'name': 'squared_error', 'masked': True}
    >>> resolve_loss({"masked": True}) == Loss(masked=True)
    True

    """
    name: str = DEFAULT_LOSS
    masked: bool = False

    def __post_init__(self):
        # Fail at configuration rather than at fit or checkpoint load.
        self.function

    @cached_property
    def function(self) -> Callable:
        """Elementwise loss returning value, gradient and Hessian."""
        from ..config.instantiate import resolve_class

        try:
            return LOSSES[self.name]
        except KeyError:
            return resolve_class(self.name)

    def to_dict(self) -> dict[str, ...]:
        return asdict(self)

    def bounds(
        self,
        y: ArrayLike,
        censor: ArrayLike | None = None,
    ):
        """Return lower and upper bounds on the true target values.

        ``censor`` holds indicators shaped like ``y``: negative for
        left-censored (true value at most the target), positive for
        right-censored (at least the target) and zero for exact targets.
        """
        import numpy as np

        y = np.asarray(y, dtype=float)
        lower, upper = y.copy(), y.copy()
        if censor is not None:
            direction = np.sign(np.reshape(censor, y.shape))
            lower[direction < 0] = -np.inf
            upper[direction > 0] = np.inf
        if self.masked:
            missing = ~np.isfinite(y)
            lower[missing], upper[missing] = -np.inf, np.inf
        return lower, upper

    def requires_objective(
        self, 
        y: ArrayLike, 
        censor: ArrayLike | None = None,
    ) -> bool:
        """Whether training needs this loss as a custom objective."""
        import numpy as np

        return (
            self.name != DEFAULT_LOSS
            or censor is not None
            or (self.masked and not np.all(np.isfinite(y)))
        )

    def gradients(self, prediction, lower, upper, weight=None):
        """Weighted gradient and Hessian with respect to ``prediction``."""
        _, gradient, hessian = self.function(prediction, lower, upper)
        weight = _row_weights(weight, gradient.shape)
        return gradient * weight, hessian * weight

    def mean(self, prediction, lower, upper, weight=None) -> float:
        """Weighted mean loss over unmasked targets."""
        import numpy as np

        value, _, _ = self.function(prediction, lower, upper)
        weight = _row_weights(weight, value.shape) * (
            np.isfinite(lower) | np.isfinite(upper)
        )
        return float(np.sum(value * weight) / np.sum(weight))


def resolve_loss(
    loss: str | Mapping[str, ...] | Callable | Loss | None = None,
) -> Loss | None:
    """Resolve a loss name, field mapping, importable function, or Loss."""
    if loss is None or isinstance(loss, Loss):
        return loss
    if isinstance(loss, str):
        return Loss(name=loss)
    if isinstance(loss, Mapping):
        return Loss(**loss)
    if callable(loss):
        return Loss(name=f"{loss.__module__}.{loss.__qualname__}")
    raise TypeError(
        "loss must be a name, mapping, importable function or Loss, "
        f"but was {type(loss).__name__}: {loss}"
    )
