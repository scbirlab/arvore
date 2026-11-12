"""Shared access to fitted training-reference data."""

from collections.abc import Mapping


def training_data(box):
    """Resolve the fitted training data retained by Box."""
    return box._training_reference()


def iter_batches(data, batch_size):
    """Re-iterable Dataset or column-mapping batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if hasattr(data, "iter"):
        yield from data.iter(batch_size=batch_size)
        return

    if isinstance(data, Mapping):
        lengths = {len(value) for value in data.values()}
        if len(lengths) != 1:
            raise ValueError("Reference column lengths differ")

        n = lengths.pop()
        for start in range(0, n, batch_size):
            yield {
                key: value[start:start + batch_size]
                for key, value in data.items()
            }
        return

    raise TypeError(
        "Training reference must be a Dataset "
        "or a mapping of columns"
    )


def feature_batches(box, *, batch_size):
    """Yield model-space feature matrices from fitted training data."""
    from ..box.base import _matrix

    data = training_data(box)

    for batch in iter_batches(data, batch_size):
        if not batch:
            continue

        (X,) = _matrix(
            batch,
            box.pipeline,
            (box.features,),
        )

        yield X
