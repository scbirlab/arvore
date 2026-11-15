"""Checkpoint fitted models through their serialization adapters."""

from collections.abc import Mapping
from hashlib import sha256
import os
from pathlib import Path

from ...config import ModelArtifact
from .base import _backend_version, _class_path
from .dispatch import get_adapter_by_name, get_serialization_adapter
from .sklearn import _skops_digest


def save_model(
    model, 
    path: str | os.PathLike, 
    *,
    skip_save: bool = False,
    **kwargs
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapter = get_serialization_adapter(
        model,
        allow_unsafe=kwargs.get("allow_unsafe", False),
    )
    path = path / adapter.filename
    if skip_save:
        trainable = True
    else:
        adapter.save(model, path, **kwargs)
        if adapter.name == "ensemble":
            from providence.io import load_json

            manifest = load_json(path)
            trainable = all(
                entry["metadata"]["trainable"]
                for entry in manifest["members"]
            )
        else:
            trainable = adapter.restored_trainable

    return ModelArtifact(
        adapter=adapter.name,
        filename=adapter.filename if not skip_save else None,
        class_path=_class_path(model),
        version=_backend_version(adapter.name),
        sha256=(
            sha256(path.read_bytes()).hexdigest()
            if not skip_save else None
        ),
        trainable=trainable,
    ).to_dict()


def model_digest(model) -> str:
    """Content identity of a fitted model, stable across save and load.

    skops models get a canonical content digest; other backends the
    SHA-256 of their native serialization. Raises TypeError for
    estimators without a safe codec.
    """
    import json
    import tempfile
    from ..ensemble import Ensemble

    if isinstance(model, Ensemble):
        return sha256(json.dumps({
            "weights": [float(w) for w in model.weights],
            "members": {
                name: model_digest(member)
                for name, member in model.members.items()
            },
        }, sort_keys=True).encode("utf-8")).hexdigest()

    adapter = get_serialization_adapter(model, allow_unsafe=False)
    if adapter.name == "skops":
        import skops.io as sio
        return _skops_digest(sio.dumps(model))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / adapter.filename
        adapter.save(model, path)
        return sha256(path.read_bytes()).hexdigest()


def load_model(
    path: str | os.PathLike,
    artifact: ModelArtifact | Mapping,
    *,
    allow_unsafe: bool = False,
    trusted_types=(),
    **kwargs
):
    if not isinstance(artifact, ModelArtifact):
        artifact = ModelArtifact.from_dict(artifact)

    path = Path(path)
    name = artifact.adapter

    if name == "joblib" and not allow_unsafe:
        raise ValueError(
            "Joblib checkpoints require allow_unsafe=True"
        )

    adapter = get_adapter_by_name(name)
    filename = artifact.filename
    # ModelArtifact sets filename and sha256 together, or neither.
    if filename is None:
        raise ValueError("Cannot load a model from a checkpoint saved with skip_save=True")
    if filename != adapter.filename:
        raise ValueError(
            f"Unexpected filename for adapter {name}: {filename}"
        )

    path = path / filename
    digest = sha256(path.read_bytes()).hexdigest()

    if digest != artifact.sha256:
        raise ValueError("Model checkpoint checksum mismatch")

    return adapter.load(
        path, 
        trusted=tuple(trusted_types),
        allow_unsafe=allow_unsafe,
        **kwargs
    )
