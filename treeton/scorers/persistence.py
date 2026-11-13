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
CODEC_TYPES = frozenset({
    "leaf_counts", "leaf_percentile", "euclidean", "tanimoto",
})


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


def _encode_leaf(value, calibrated):
    from .leaves import LeafReference

    if not isinstance(value, tuple) or not value:
        raise ValueError("Leaf reference must be a nonempty tuple")
    arrays = {"n_members": np.array(len(value), dtype=np.int64)}
    for i, ref in enumerate(value):
        if not isinstance(ref, LeafReference) or not ref.ids:
            raise ValueError("Invalid leaf reference")
        lengths = [len(ids) for ids in ref.ids]
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        offsets[1:] = np.cumsum(lengths)
        arrays[f"offsets_{i}"] = offsets
        arrays[f"ids_{i}"] = np.concatenate(ref.ids).astype(np.int64)
        arrays[f"counts_{i}"] = np.concatenate(ref.counts).astype(np.int64)
        arrays[f"n_reference_{i}"] = np.array(ref.n_reference, dtype=np.int64)
        if calibrated:
            if ref.calibration_scores is None:
                raise ValueError("Missing leaf calibration scores")
            arrays[f"calibration_{i}"] = np.asarray(
                ref.calibration_scores, dtype=np.float64
            )
        elif ref.calibration_scores is not None:
            raise ValueError("Raw leaf-count reference contains calibration")
    return arrays


def _decode_leaf(archive, calibrated):
    from .leaves import LeafReference

    n_members = int(archive["n_members"])
    if n_members < 1:
        raise ValueError("Invalid leaf reference member count")
    references = []
    for i in range(n_members):
        offsets = np.asarray(archive[f"offsets_{i}"])
        ids = np.asarray(archive[f"ids_{i}"])
        counts = np.asarray(archive[f"counts_{i}"])
        n_reference = int(archive[f"n_reference_{i}"])
        if (
            offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer)
            or len(offsets) < 2 or offsets[0] != 0
            or np.any(np.diff(offsets) <= 0)
            or ids.ndim != 1 or counts.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(counts.dtype, np.integer)
            or int(offsets[-1]) != len(ids) or len(ids) != len(counts)
            or n_reference < 1
        ):
            raise ValueError("Invalid saved leaf-reference arrays")
        tree_ids = tuple(
            np.asarray(ids[a:b], dtype=np.int64).copy()
            for a, b in zip(offsets[:-1], offsets[1:])
        )
        tree_counts = tuple(
            np.asarray(counts[a:b], dtype=np.int64).copy()
            for a, b in zip(offsets[:-1], offsets[1:])
        )
        for one_ids, one_counts in zip(tree_ids, tree_counts):
            if (
                np.any(np.diff(one_ids) <= 0)
                or np.any(one_counts <= 0)
                or int(one_counts.sum()) != n_reference
            ):
                raise ValueError("Invalid saved leaf occupancy counts")
        ref = LeafReference(tree_ids, tree_counts, n_reference)
        if calibrated:
            scores = np.asarray(archive[f"calibration_{i}"], dtype=float)
            if scores.shape != (n_reference,) or not np.all(np.isfinite(scores)):
                raise ValueError("Invalid saved leaf calibration")
            if np.any(np.diff(scores) < 0):
                raise ValueError("Leaf calibration scores must be sorted")
            ref = ref.with_calibration(scores)
        references.append(ref)
    return tuple(references)


def _encode(resource_type, value):
    if resource_type in {"leaf_counts", "leaf_percentile"}:
        return _encode_leaf(value, resource_type == "leaf_percentile")
    if resource_type == "euclidean":
        from .distances import EuclideanReference
        if not isinstance(value, EuclideanReference):
            raise TypeError("Expected EuclideanReference")
        return {"features": np.asarray(value.features, dtype=np.float64)}
    if resource_type == "tanimoto":
        from .tanimoto import TanimotoReference
        if not isinstance(value, TanimotoReference):
            raise TypeError("Expected TanimotoReference")
        return {
            "packed": np.asarray(value.packed, dtype=np.uint8),
            "counts": np.asarray(value.counts, dtype=np.int32),
            "n_bits": np.array(value.n_bits, dtype=np.int64),
        }
    raise ValueError(f"No safe reference codec for {resource_type!r}")


def _decode(resource_type, archive):
    if resource_type in {"leaf_counts", "leaf_percentile"}:
        return _decode_leaf(archive, resource_type == "leaf_percentile")

    if resource_type == "euclidean":
        from .distances import EuclideanReference
        features = np.asarray(archive["features"])
        if features.dtype != np.float64:
            raise ValueError("Expected float64 Euclidean features")
        return EuclideanReference.build(features)

    if resource_type == "tanimoto":
        from .tanimoto import TanimotoReference
        packed = np.asarray(archive["packed"])
        counts = np.asarray(archive["counts"])
        n_bits = int(archive["n_bits"])
        if (
            packed.ndim != 2 or len(packed) < 1
            or packed.dtype != np.uint8
            or counts.ndim != 1 or len(counts) != len(packed)
            or not np.issubdtype(counts.dtype, np.integer)
            or n_bits < 1 or packed.shape[1] != (n_bits + 7) // 8
            or np.any(
                counts != np.bitwise_count(packed).sum(
                    axis=1,
                    dtype=np.int32,
                )
            )

        ):
            raise ValueError("Invalid Tanimoto reference archive")
        if n_bits % 8:
            padding_mask = 255 ^ ((1 << (n_bits % 8)) - 1)
            if np.any(packed[:, -1] & padding_mask):
                raise ValueError("Nonzero fingerprint padding bits")
        return TanimotoReference(
            packed.copy(), counts.astype(np.int32, copy=True), n_bits
        )
    raise ValueError(f"Unsupported reference type: {resource_type!r}")


def save_references(checkpoint, entries, state_digest):
    """Write all supported prepared resources; return manifest metadata."""
    stored = []
    for key, entry in sorted(entries.items()):
        spec, value = entry
        spec = canonical_spec(spec)
        if reference_key(spec) != key:
            raise ValueError("Reference cache key does not match specification")

        kind = spec["type"]
        if kind not in CODEC_TYPES:
            continue  # Optional in-memory-only reference.

        if spec.get("version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported {kind!r} reference spec version")

        directory = Path(checkpoint) / "references"
        directory.mkdir(exist_ok=True)
        filename = f"{key}.npz"
        file = directory / filename
        np.savez_compressed(file, **_encode(kind, value))
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


def load_references(
    box, 
    checkpoint, 
    metadata, 
    config
):
    """Verify *all* archives before installing any cached resource."""
    if metadata is None:
        return

    if not isinstance(metadata, dict) or metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported reference checkpoint manifest")

    if metadata.get("state_sha256") != fitted_state_digest(checkpoint, config):
        raise ValueError("Reference fitted-state checksum mismatch")

    items = metadata.get("items")
    if not isinstance(items, list):
        raise ValueError("Invalid reference checkpoint entries")

    recovered = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid reference checkpoint item")
        spec = canonical_spec(item["spec"])
        key = reference_key(spec)
        if (
            key != item.get("key")
            or spec["type"] not in CODEC_TYPES
            or spec.get("version") != FORMAT_VERSION
            or item.get("format_version") != FORMAT_VERSION
            or item.get("filename") != f"{key}.npz"
            or key in recovered
        ):
            raise ValueError("Incompatible reference resource metadata")
        file = Path(checkpoint) / "references" / f"{key}.npz"
        if _sha256(file) != item.get("sha256"):
            raise ValueError("Reference checksum mismatch")
        with np.load(file, allow_pickle=False) as archive:
            value = _decode(spec["type"], archive)
        recovered[key] = (spec, value)
    box.references.restore(recovered)
    return None


def validate_requested(box, scorers, entries):
    """Fail instead of silently omitting explicitly requested resources."""
    for name, scorer in scorers.items():
        hook = getattr(scorer, "reference_specs", None)
        if not callable(hook):
            continue
        for spec in hook(box):
            spec = canonical_spec(spec)
            key = reference_key(spec)
            if spec["type"] not in CODEC_TYPES or spec.get("version") != FORMAT_VERSION:
                raise ValueError(
                    f"Scorer {name!r} requested a reference without a "
                    f"safe persistence codec: {spec['type']!r}"
                )
            if key not in entries:
                raise ValueError(
                    f"Scorer {name!r} did not prepare declared reference {key}"
                )
