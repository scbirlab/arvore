from typing import TYPE_CHECKING, Any, TypeAlias, Protocol, runtime_checkable
from collections.abc import Callable, Mapping

from providence.typing import DataLike
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from aspect.data import DataPipeline
    from .models.losses import Loss
else:
    DataPipeline = Any
    Loss = Any

PipelineLike: TypeAlias = DataPipeline | Mapping[str, Any] | str | Callable
Validation: TypeAlias = (
    tuple[ArrayLike, ArrayLike]
    | tuple[ArrayLike, ArrayLike, ArrayLike | None]
    | None
)

@runtime_checkable
class Estimator(Protocol):
    def fit(X, y, **kwargs):
        ...

    def predict(X, *kwargs):
        ...


class FitAdapter(Protocol):
    def fit(
        self,
        estimator: Estimator,
        X: ArrayLike,
        y: ArrayLike,
        *,
        validation: Validation | None = None,
        fit_kwargs: Mapping[str, ...] | None = None,
        early_stopping_rounds: int | None = None,
        loss: Loss | None = None,
        censor: ArrayLike | None = None,
    ) -> None:
        ...

    def predict(
        self, 
        estimator: Estimator, 
        X: ArrayLike,
        **kwargs
    ) -> ArrayLike:
        ...
