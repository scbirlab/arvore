"""Columns and numeric matrices from Datasets, column mappings and Aspect pipelines."""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterable, Mapping
import os

import numpy as np

if TYPE_CHECKING:
    from aspect import DataPipeline
    from datasets import Dataset, IterableDataset
else:
    DataPipeline = Any
    Dataset, IterableDataset = Any, Any

from .typing import PipelineLike


def is_dataset(data) -> bool:
    from datasets import Dataset
    return isinstance(data, Dataset)


def unwrap(data):
    """Return the data held by a Providence DataResolver, or ``data`` itself."""
    from providence import DataResolver
    return data.data if isinstance(data, DataResolver) else data


def resolve_pipeline(
    pipeline: PipelineLike | None = None,
    *,
    cache_dir: str | None = None,
) -> DataPipeline:
    """Resolve an Aspect pipeline or its declarative configuration."""
    from aspect import DataPipeline

    if isinstance(pipeline, DataPipeline):
        return pipeline

    if pipeline is None:
        pipeline = {}

    if isinstance(pipeline, os.PathLike):
        pipeline = os.fspath(pipeline)

    if isinstance(pipeline, (str, Mapping)):
        return DataPipeline.from_config(
            pipeline,
            cache_dir=cache_dir,
        )

    raise TypeError(
        "pipeline must be an Aspect DataPipeline, "
        "a configuration mapping, or a JSON filename; "
        f"received {type(pipeline).__name__}"
    )


def column_names(data: Dataset | IterableDataset | Mapping) -> list[str]:
    from datasets import Dataset, IterableDataset

    if isinstance(data, (Dataset, IterableDataset)):
        return data.column_names
    if isinstance(data, Mapping):
        return list(data)
    raise ValueError(f"Can't get column names for {type(data)}: {data}")


def _as_numeric(values: np.ndarray) -> np.ndarray:
    """Booleans as 0/1, and missing values in bool or numeric columns as NaN.

    Other dtypes are returned unchanged.

    Examples
    ========
    >>> _as_numeric(np.array([True, False]))
    array([1, 0], dtype=uint8)
    >>> _as_numeric(np.array([True, None, 2.5], dtype=object))
    array([1. , nan, 2.5])
    >>> _as_numeric(np.array(["1", None], dtype=object)).dtype
    dtype('O')

    """
    if values.dtype == bool:
        return values.astype(np.uint8)
    if values.dtype == object:
        import pandas as pd
        missing = pd.isna(values)
        present = values[~missing]
        if all(isinstance(v, (bool, int, float, np.bool_, np.number)) for v in present):
            numeric = np.full(values.shape, np.nan)
            numeric[~missing] = present.astype(float)
            return numeric
    return values


def matrix(
    data: Dataset | Mapping,
    column_groups: Iterable[Iterable[str]]
) -> tuple[np.ndarray, ...]:
    """Stack each group of columns into a 2D numeric array.

    Transformed columns are looked up by their output names.

    Boolean columns become 0/1. Missing values in boolean or numeric
    columns, including nullable booleans, become NaN.

    """
    blocks = ()
    available = column_names(data)
    for columns in column_groups:
        arrays = []
        for column in columns:
            if column not in available:
                raise KeyError(
                    f"Column or Aspect transform {column} unavailable. "
                    f"Available columns: {available}"
                )
            values = data[column]
            if len(values) == 0:
                raise ValueError(f"No rows in dataset: {data}")
            try:
                values = np.asarray(values)
                if values.dtype == object:
                    values = np.stack(list(values))
                values = _as_numeric(values.reshape(len(values), -1))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Column {column!r} is not a regular array: {data[column]}") from error
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(f"Column {column!r} must be numeric, got {values.dtype}")
            arrays.append(values)
        blocks += (np.concatenate(arrays, axis=1),)
    return blocks
