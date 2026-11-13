"""Versioned, pickle-free persistence for prepared scorer references.

Cache identity is scoped to a fitted Box. A checkpoint binds references to
its exact model, preprocessing and saved training-data metadata.
"""

from hashlib import sha256
from pathlib import Path
import json
import os

import numpy as np

from .references import canonical_spec, reference_key


FORMAT_VERSION: int = 1


def _sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fitted_state_digest(checkpoint, config):
    """Hash saved model, fitted preprocessing and Box config.

    Exclude only the reference archive and the top-level manifest to avoid a
    circular dependency. Hash symbolic link targets without following them.
    """
    checkpoint = Path(checkpoint)
    digest = sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))
    for path in sorted(checkpoint.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(checkpoint).as_posix()
        if relative == "config.json" or relative.startswith("references/"):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"SYMLINK:")
            digest.update(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            digest.update(bytes.fromhex(_sha256(path)))
        else:
            raise ValueError(f"Unsupported checkpoint item: {relative}")
    return digest.hexdigest()



def reference_types():
    """Registry of supported safe reference formats."""
    from .leaves import LeafReferences
    from .distances import EuclideanReference
    from .tanimoto import TanimotoReference

    return {
        "leaf_counts": LeafReferences,
        "leaf_percentile": LeafReferences,
        "euclidean": EuclideanReference,
        "tanimoto": TanimotoReference,
    }


def validate_requested(requested, entries):
    """Explicitly requested references must be persistable."""
    types = reference_types()

    for key in requested:
        if key not in entries:
            raise ValueError(
                f"Requested reference {key} is missing"
            )

        spec, value = entries[key]
        kind = spec["type"]
        cls = types.get(kind)

        if (
            cls is None
            or spec.get("version") != FORMAT_VERSION
            or not isinstance(value, cls)
        ):
            raise ValueError(
                f"Requested reference {kind!r} has no "
                "safe persistence codec"
            )


def save_references(checkpoint, entries, state_digest):
    """Persist supported reference state dictionaries."""
    types = reference_types()
    stored = []

    for key, (spec, value) in sorted(entries.items()):
        spec = canonical_spec(spec)

        if reference_key(spec) != key:
            raise ValueError(
                "Reference cache key does not match specification"
            )

        kind = spec["type"]
        cls = types.get(kind)

        # In-memory-only references may remain uncached on disk.
        # Explicit save requests are checked separately.
        if cls is None:
            continue

        if spec.get("version") != FORMAT_VERSION:
            raise ValueError(
                "Unsupported reference specification version"
            )

        if not isinstance(value, cls):
            raise TypeError(
                f"Reference {kind!r} should be {cls.__name__}"
            )

        state = value.state_dict()

        if not isinstance(state, dict) or not all(
            isinstance(name, str)
            for name in state
        ):
            raise TypeError(
                "Reference state must be a string-keyed dict"
            )

        arrays = {
            name: np.asarray(array)
            for name, array in state.items()
        }

        if any(
            array.dtype.hasobject
            for array in arrays.values()
        ):
            raise TypeError(
                "Object arrays cannot be checkpointed safely"
            )

        directory = Path(checkpoint) / "references"
        directory.mkdir(exist_ok=True)

        filename = f"{key}.npz"
        file = directory / filename

        np.savez_compressed(file, **arrays)

        stored.append({
            "key": key,
            "spec": spec,
            "filename": filename,
            "format_version": FORMAT_VERSION,
            "sha256": _sha256(file),
        })

    if not stored:
        return None

    return {
        "format_version": FORMAT_VERSION,
        "state_sha256": state_digest,
        "items": stored,
    }


def load_references(box, checkpoint, metadata, config):
    """Verify all resources before installing any of them."""
    if metadata is None:
        return

    if (
        not isinstance(metadata, dict)
        or metadata.get("format_version")
        != FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported reference checkpoint manifest"
        )

    if metadata.get("state_sha256") != fitted_state_digest(
        checkpoint, config
    ):
        raise ValueError(
            "Reference fitted-state checksum mismatch"
        )

    items = metadata.get("items")
    if not isinstance(items, list):
        raise ValueError(
            "Invalid reference checkpoint entries"
        )

    types = reference_types()
    recovered = {}

    for item in items:
        if not isinstance(item, dict):
            raise ValueError(
                "Invalid reference checkpoint item"
            )

        spec = canonical_spec(item["spec"])
        key = reference_key(spec)
        kind = spec["type"]
        cls = types.get(kind)

        if (
            key != item.get("key")
            or cls is None
            or spec.get("version") != FORMAT_VERSION
            or item.get("format_version") != FORMAT_VERSION
            or item.get("filename") != f"{key}.npz"
            or key in recovered
        ):
            raise ValueError(
                "Incompatible reference resource metadata"
            )

        file = (
            Path(checkpoint)
            / "references"
            / f"{key}.npz"
        )

        if _sha256(file) != item.get("sha256"):
            raise ValueError(
                "Reference checksum mismatch"
            )

        with np.load(file, allow_pickle=False) as archive:
            value = cls.from_state_dict(archive, spec)

        recovered[key] = (spec, value)

    box.references.restore(recovered)
    return None
