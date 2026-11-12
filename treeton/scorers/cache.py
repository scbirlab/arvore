"""Per-Box scorer cache and pickle-free leaf-reference persistence."""

from hashlib import sha256
from pathlib import Path

import numpy as np

from ..models.ensemble import Ensemble
from .leaves import LeafReference


FILENAME = "leaf-reference.npz"
FORMAT_VERSION = 1


def _members(box):
    if isinstance(box.model, Ensemble):
        return tuple(box.model.members.values())
    return (box.model,)


def _token(box):
    """Detect replacement of cache-dependent Box objects."""
    pipeline = box.pipeline

    return (
        id(box.model),
        tuple(id(member) for member in _members(box)),
        id(pipeline),
        id(getattr(pipeline, "data_in", None)),
        id(getattr(pipeline, "data_out", None)),
        tuple(box.features),
    )


def get_score_cache(box):
    """Return the cache for the current fitted Box state."""
    token = _token(box)
    cache = getattr(box, "_scorer_cache", None)

    if not isinstance(cache, dict) or cache.get("_token") != token:
        cache = {
            "_token": token,
            "leaf_reference": None,
            "leaf_percentile": None,
        }
        box._scorer_cache = cache

    return cache


def invalidate_score_cache(box):
    """Clear cached diagnostics before fitting a model."""
    box._scorer_cache = None


def _digest(path):
    hash_value = sha256()

    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            hash_value.update(block)

    return hash_value.hexdigest()


def save_score_cache(box, checkpoint_path):
    """Persist a warmed leaf-reference cache, if present.

    Returns checkpoint manifest metadata, or None.
    """
    checkpoint_path = Path(checkpoint_path)
    output = checkpoint_path / FILENAME
    cache = getattr(box, "_scorer_cache", None)

    if (
        not isinstance(cache, dict)
        or cache.get("_token") != _token(box)
        or cache.get("leaf_reference") is None
    ):
        output.unlink(missing_ok=True)
        return None

    references = cache["leaf_reference"]
    calibrated = cache.get("leaf_percentile")

    arrays = {
        "version": np.array(FORMAT_VERSION, dtype=np.int64),
        "n_members": np.array(len(references), dtype=np.int64),
    }

    if len(references) != len(_members(box)):
        raise ValueError("Cached member count does not match model")

    if calibrated is not None and len(calibrated) != len(references):
        raise ValueError("Cached calibration count does not match")

    for i, reference in enumerate(references):
        if not reference.ids:
            raise ValueError("Reference has no trees")

        lengths = [len(ids) for ids in reference.ids]
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        offsets[1:] = np.cumsum(lengths)

        arrays[f"offsets_{i}"] = offsets
        arrays[f"ids_{i}"] = np.concatenate(reference.ids)
        arrays[f"counts_{i}"] = np.concatenate(reference.counts)
        arrays[f"n_reference_{i}"] = np.array(
            reference.n_reference,
            dtype=np.int64,
        )

        if calibrated is not None:
            scores = calibrated[i].calibration_scores

            if scores is None:
                raise ValueError(
                    "Incomplete cached percentile calibration"
                )

            arrays[f"calibration_{i}"] = scores

    np.savez_compressed(output, **arrays)

    return {
        "filename": FILENAME,
        "sha256": _digest(output),
    }


def load_score_cache(box, checkpoint_path, metadata):
    """Restore verified leaf references without unpickling."""
    if metadata is None:
        return

    if (
        not isinstance(metadata, dict)
        or metadata.get("filename") != FILENAME
        or not isinstance(metadata.get("sha256"), str)
    ):
        raise ValueError("Invalid scorer-cache manifest entry")

    path = Path(checkpoint_path) / FILENAME

    if _digest(path) != metadata["sha256"]:
        raise ValueError("Leaf-reference cache checksum mismatch")

    references = []
    calibrated = []
    calibration_present = []

    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["version"])
        n_members = int(archive["n_members"])

        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported scorer-cache format: {version}"
            )

        if n_members != len(_members(box)) or n_members < 1:
            raise ValueError(
                "Scorer-cache member count does not match model"
            )

        for i in range(n_members):
            offsets = np.asarray(
                archive[f"offsets_{i}"],
                dtype=np.int64,
            )
            ids = np.asarray(
                archive[f"ids_{i}"],
                dtype=np.int64,
            )
            counts = np.asarray(
                archive[f"counts_{i}"],
                dtype=np.int64,
            )
            n_reference = int(
                archive[f"n_reference_{i}"]
            )

            if (
                n_reference < 1
                or offsets.ndim != 1
                or len(offsets) < 2
                or offsets[0] != 0
                or offsets[-1] != len(ids)
                or len(ids) != len(counts)
                or np.any(np.diff(offsets) <= 0)
            ):
                raise ValueError("Invalid cached leaf-reference shape")

            trees_ids = tuple(
                ids[start:stop].copy()
                for start, stop in zip(
                    offsets[:-1], offsets[1:]
                )
            )
            trees_counts = tuple(
                counts[start:stop].copy()
                for start, stop in zip(
                    offsets[:-1], offsets[1:]
                )
            )

            for tree_ids, tree_counts in zip(
                trees_ids, trees_counts
            ):
                if (
                    np.any(np.diff(tree_ids) <= 0)
                    or np.any(tree_counts <= 0)
                    or int(tree_counts.sum()) != n_reference
                ):
                    raise ValueError("Invalid cached leaf counts")

            reference = LeafReference(
                ids=trees_ids,
                counts=trees_counts,
                n_reference=n_reference,
            )
            references.append(reference)

            key = f"calibration_{i}"
            present = key in archive.files
            calibration_present.append(present)

            if present:
                calibrated.append(
                    reference.with_calibration(
                        archive[key]
                    )
                )

    if any(calibration_present) and not all(calibration_present):
        raise ValueError("Incomplete leaf calibration archive")

    cache = get_score_cache(box)
    cache["leaf_reference"] = tuple(references)

    if all(calibration_present):
        cache["leaf_percentile"] = tuple(calibrated)
