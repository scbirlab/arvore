"""Content identities of fitted Boxes, their data and the operations on them."""

from ..tabular import is_dataset


def pipeline_identity(pipeline):
    """Fitted preprocessing identity from its config, never its data."""
    from datasets.fingerprint import Hasher

    if pipeline is None:
        return None
    if callable(getattr(pipeline, "to_config", None)):
        return pipeline.to_config()
    return Hasher.hash(pipeline)


def data_identity(data):
    """Content checksum of retained data, independent of storage layout."""
    from pandas import DataFrame
    from providence.io import dataframe_checksum

    if data is None:
        return None
    if is_dataset(data):
        return dataframe_checksum(data.with_format(None).to_pandas())
    return dataframe_checksum(DataFrame(dict(data)))


def model_identity(model) -> str:
    """Content digest of a fitted model, stable across save and load.

    Estimators without a safe codec fall back to an in-process hash,
    which is not stable across load.
    """
    from datasets.fingerprint import Hasher
    from ..models.adapters import model_digest

    try:
        return model_digest(model)
    except TypeError:
        return "in-process:" + Hasher.hash(model)


def box_fingerprint(box) -> str:
    from datasets.fingerprint import Hasher
    from aspect import __version__ as aspect_version
    from .. import __version__

    try:
        training = data_identity(box.training_data)
    except (AttributeError, ValueError):
        training = None

    return Hasher.hash({
        "arvore": __version__,
        "aspect": aspect_version,
        "model": model_identity(box.model),
        "pipeline": pipeline_identity(box.pipeline),
        "features": list(box.features),
        "target": list(box.target),
        "training": training,
    })


def operation_fingerprint(box, dataset, operation, **params) -> str:
    """Fingerprint for one Box operation over one input Dataset."""
    from datasets.fingerprint import Hasher
    from ..prediction import predict_map_batch

    return Hasher.hash({
        "input": dataset._fingerprint,
        "operation": operation,
        "function": Hasher.hash(predict_map_batch),
        "box": box.fingerprint,
        **params,
    })
