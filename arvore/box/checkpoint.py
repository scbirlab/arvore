"""Atomic, pickle-free Box checkpoints."""

from collections.abc import Iterable
from pathlib import Path
import os
import warnings

from providence.io import save_json, load_json

from .base import DEFAULT_BATCH_SIZE

CONFIG_FILENAME: str = "config.json"
SCHEMA_VERSION: int = 2


def save(
    box,
    path: str | os.PathLike,
    allow_unsafe: bool = False,
    *,
    score=None,
    save_references: bool = True,
    **pipeline_kwargs,
) -> None:
    import shutil
    import tempfile
    import uuid

    from ..models.adapters import save_model
    from ..scorers.registry import resolve_scorers
    from ..scorers.runner import ScoringRunner
    from ..scorers.persistence import (
        fitted_state_digest,
        save_references as persist_references,
        validate_requested,
    )

    if not isinstance(save_references, bool):
        raise TypeError("save_references must be a boolean")

    if save_references:
        scorers = resolve_scorers(score)
        with box.references.track() as requested:
            if scorers:
                ScoringRunner(
                    box, scorers
                ).prepare(batch_size=DEFAULT_BATCH_SIZE)

        snapshot = box.references.snapshot()
        validate_requested(requested, snapshot)
    else:
        # Skip scorer resolution, preparation and persistence.
        # Existing in-memory references remain untouched.
        snapshot = {}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{path.name}.staging-", dir=path.parent,
    ))
    try:
        model_metadata = save_model(
            box.model, staging, allow_unsafe=allow_unsafe,
        )
        config = box.to_config()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "model": model_metadata,
            "training_identity": box._training_identity,
            "artefacts": {"training_data": "training-data.as"},
        }
        box.pipeline.save(
            staging / manifest["artefacts"]["training_data"],
            **pipeline_kwargs,
        )
        if save_references:
            identity = fitted_state_digest(staging, config)
            refs = persist_references(staging, snapshot, identity)
            if refs is not None:
                manifest["artefacts"]["references"] = refs
        save_json(manifest, staging / CONFIG_FILENAME)

        # Staging is complete before publishing. Directory overwrite
        # uses a rollback backup; never publish a partial checkpoint.
        backup = None
        if path.exists():
            if not path.is_dir():
                raise FileExistsError(path)
            backup = path.with_name(
                f".{path.name}.backup-{uuid.uuid4().hex}"
            )
            path.rename(backup)
        try:
            staging.rename(path)
        except BaseException:
            if backup is not None and not path.exists():
                backup.rename(path)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return None


def load(
    cls,
    path: str | os.PathLike,
    allow_unsafe: bool = False,
    *,
    trusted_types: Iterable[str] | None = None,
    cache_dir: str | None = None,
):
    from aspect.data import DataPipeline
    from ..models.adapters import load_model
    import joblib

    path = Path(path)
    manifest = load_json(path / CONFIG_FILENAME)
    schema = manifest["schema_version"]
    if trusted_types is None:
        trusted_types = ()
    if schema == 1:
        warnings.warn(
            "Schema-1 (joblib) checkpoints are deprecated and will not load "
            "in a future release. Save the Box again to upgrade it.",
            DeprecationWarning,
            stacklevel=3,
        )
        if not allow_unsafe:
            raise ValueError(
                "Legacy schema-1 checkpoints use joblib. "
                "Pass allow_unsafe=True only for trusted files."
            )
        model = joblib.load(
            path / manifest["artefacts"]["model_binary"]
        )

    else:
        if schema != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Arvore checkpoint schema: "
                f"{manifest['schema_version']}, but package is schema version {SCHEMA_VERSION}"
            )
        model = load_model(
            path,
            manifest["model"],
            allow_unsafe=allow_unsafe,
            trusted_types=trusted_types,
        )
    artefacts = manifest["artefacts"]
    config = manifest["config"]
    pipeline = DataPipeline.load(
        path / artefacts["training_data"],
        cache_dir=cache_dir,
    )
    box = cls(
        model=model,
        pipeline=pipeline,
        features=config["features"],
        target=config["target"],
        model_config=config.get("model"),
        censor=config.get("censor"),
        loss=config.get("loss"),
    )
    # Loading the pipeline has verified the training data: embedded data
    # against the checksum saved with it, local data saved by reference
    # against its source checksum, and remote data by its pinned commit.
    # The identity taken at fit time therefore still holds, whereas in-memory
    # data would otherwise be re-identified by the embedded copy's new
    # fingerprint. The manifest itself is not signed: like the rest of a
    # checkpoint, the identity is only as trustworthy as its source.
    if "training_identity" in manifest:
        box._training_identity = manifest["training_identity"]

    from ..scorers.persistence import load_references
    load_references(
        box, path, artefacts.get("references"), config,
    )

    return box
