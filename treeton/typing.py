from typing import TYPE_CHECKING, Any, TypeAlias
from collections.abc import Callable, Mapping

if TYPE_CHECKING:
    from aspect.data import DataPipeline
else:
    DataPipeline = Any

PipelineLike: TypeAlias = DataPipeline | Mapping[str, Any] | str | Callable
