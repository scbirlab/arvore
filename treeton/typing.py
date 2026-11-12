from typing import TYPE_CHECKING, Any, TypeAlias, Protocol, runtime_checkable
from collections.abc import Callable, Mapping

from providence.typing import DataLike

if TYPE_CHECKING:
    from aspect.data import DataPipeline
else:
    DataPipeline = Any

PipelineLike: TypeAlias = DataPipeline | Mapping[str, Any] | str | Callable

@runtime_checkable
class Estimator(Protocol):
    def fit(X, y, **kwargs):
        ...

    def predict(X, *kwargs):
        ...
