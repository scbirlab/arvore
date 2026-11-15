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


def get_transformed_column(
    column: str,
    data: Dataset | IterableDataset | Mapping,
    pipeline: DataPipeline
) -> str:
    """Name of the column holding ``column`` after the pipeline's transforms."""
    transforms = pipeline.column_transforms.get(column, ())
    available = column_names(data)
    if len(transforms) > 0:
        generated_column = transforms[-1].output_column
        if generated_column in available:
            return generated_column
    if column in available:
        return column
    raise KeyError(
        f"Column or Aspect transform {column} unavailable. "
        f"Available columns: {available}"
    )


def matrix(
    data: Dataset | Mapping,
    pipeline: DataPipeline,
    column_groups: Iterable[Iterable[str]]
) -> tuple[np.ndarray, ...]:
    """Stack each group of (transformed) columns into a 2D numeric array."""
    blocks = ()
    for columns in column_groups:
        arrays = []
        for column in columns:
            column_name = get_transformed_column(column, data, pipeline)
            values = data[column_name]
            if len(values) == 0:
                raise ValueError(f"No rows in dataset: {data}")
            try:
                values = np.asarray(values)
                if values.dtype == object:
                    values = np.stack(list(values))
                values = values.reshape(len(values), -1)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Column {column!r} is not a regular array: {data[column_name]}") from error
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(f"Column {column!r} must be numeric, got {values.dtype}")
            arrays.append(values)
        blocks += (np.concatenate(arrays, axis=1),)
    return blocks
