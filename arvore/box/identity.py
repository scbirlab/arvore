"""Content identities of fitted Boxes, their data and the operations on them."""

import json
import re
from uuid import uuid4

def pipeline_identity(pipeline):
    """Fitted preprocessing identity from its config, never its data."""
    from datasets.fingerprint import Hasher

    if pipeline is None:
        return None
    if callable(getattr(pipeline, "to_config", None)):
        return pipeline.to_config()
    return Hasher.hash(pipeline)


def _pinned(source) -> bool:
    """Whether a remote source is pinned to an immutable commit.

    Providence resolves Hugging Face revisions to commit SHAs. A branch or
    tag name, or no revision at all, can move while the URI stays the same.
    """
    return (
        source.is_remote
        and re.fullmatch(r"[0-9a-f]{40}", source.revision) is not None
    )


def training_identity(pipeline):
    """Identity of the data a pipeline was fitted on, from provenance, not contents.

    In order of preference: a remote source pinned to a commit, the checksum
    Providence records for local and in-memory sources, Datasets'
    fingerprint, or, when none exists, a unique token, so unidentifiable
    data never shares an identity. Selected rows are included; preprocessing
    is identified separately by its config.
    """
    resolver = getattr(pipeline, "data_in", None)
    if resolver is None:
        return None

    source = resolver.data_source
    fingerprint = getattr(resolver.data, "_fingerprint", None)
    if source is not None and _pinned(source):
        provenance = {"uri": source.uri, "revision": source.revision}
    elif source is not None and source.checksum is not None:
        provenance = {"checksum": source.checksum}
    elif fingerprint is not None:
        provenance = {"fingerprint": fingerprint}
    else:
        provenance = {"unidentified": uuid4().hex}
    # JSON-native, so it is unchanged by a checkpoint round trip.
    return json.loads(json.dumps({**provenance, "selected_rows": resolver.selected_rows}))


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

    return Hasher.hash({
        "arvore": __version__,
        "aspect": aspect_version,
        "model": model_identity(box.model),
        "pipeline": pipeline_identity(box.pipeline),
        "features": list(box.features),
        "target": list(box.target),
        "training": box._training_identity,
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
