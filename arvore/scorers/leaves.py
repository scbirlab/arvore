"""Reference-based tree leaf occupancy diagnostics."""

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import Requirements, ScoreContext
from .registry import register_scorer


@dataclass(frozen=True)
class LeafReference:
    """Sparse reference occupancy counts for each tree."""

    ids: tuple[NDArray, ...]
    counts: tuple[NDArray, ...]
    n_reference: int
    calibration_scores: NDArray | None = None

    def inverse_score(self, leaves: NDArray) -> NDArray:
        """Mean inverse occupancy across trees."""
        leaves = np.asarray(leaves)

        if leaves.ndim != 2:
            raise ValueError(
                "Leaf assignments must have shape (samples, trees)"
            )

        n_samples, n_trees = leaves.shape

        if n_trees != len(self.ids):
            raise ValueError(
                f"Expected {len(self.ids)} trees, "
                f"received {n_trees}"
            )

        scores = np.zeros(n_samples, dtype=float)

        for t, (ids, counts) in enumerate(
            zip(self.ids, self.counts)
        ):
            query = leaves[:, t]
            positions = np.searchsorted(ids, query)

            valid = positions < len(ids)
            matched = np.zeros(n_samples, dtype=bool)

            matched[valid] = (
                ids[positions[valid]] == query[valid]
            )

            occupancy = np.zeros(n_samples, dtype=np.int64)
            occupancy[matched] = counts[positions[matched]]

            scores += 1. / (occupancy + 1.)

        return scores / n_trees


    def with_calibration(self, scores) -> "LeafReference":
        """Attach sorted raw reference scores."""
        scores = np.asarray(scores, dtype=float)

        if scores.shape != (self.n_reference,):
            raise ValueError(
                f"Expected {self.n_reference} calibration scores, "
                f"received {scores.shape}"
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError("Calibration scores must be finite")

        return replace(
            self,
            calibration_scores=np.sort(scores),
        )

    def percentile_score(self, leaves: NDArray) -> NDArray:
        """Empirical midrank percentile of inverse occupancy."""
        reference = self.calibration_scores

        if reference is None:
            raise ValueError(
                "Reference percentile calibration is unavailable"
            )

        scores = self.inverse_score(leaves)

        left = np.searchsorted(
            reference, scores, side="left"
        )
        right = np.searchsorted(
            reference, scores, side="right"
        )

        return (left + right) / (2.0 * len(reference))


def _encode_leaf(value, calibrated):
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



class LeafReferences(tuple):
    """Immutable collection of per-member leaf references."""

    def state_dict(self):
        calibrated = all(
            ref.calibration_scores is not None
            for ref in self
        )
        return _encode_leaf(tuple(self), calibrated)

    @classmethod
    def from_state_dict(cls, state, spec):
        calibrated = spec["type"] == "leaf_percentile"
        return cls(_decode_leaf(state, calibrated))


class LeafReferenceBuilder:
    """Accumulate reference leaf counts incrementally."""

    def __init__(self):
        self._counts = None
        self.n_reference = 0

    def update(self, leaves):
        leaves = np.asarray(leaves)

        if leaves.ndim != 2:
            raise ValueError(
                "Expected (samples, trees) leaf assignments"
            )

        if not np.issubdtype(leaves.dtype, np.integer):
            raise TypeError("Leaf indices must be integers")

        if self._counts is None:
            self._counts = [
                {} for _ in range(leaves.shape[1])
            ]

        if leaves.shape[1] != len(self._counts):
            raise ValueError(
                "Number of trees changed between batches"
            )

        for t, table in enumerate(self._counts):
            ids, counts = np.unique(
                leaves[:, t],
                return_counts=True,
            )

            for leaf_id, count in zip(ids, counts):
                key = int(leaf_id)
                table[key] = table.get(key, 0) + int(count)

        self.n_reference += len(leaves)

    def finish(self) -> LeafReference:
        if not self.n_reference:
            raise ValueError("Reference dataset is empty")

        ids = []
        counts = []

        for table in self._counts:
            ordered = sorted(table)

            ids.append(
                np.asarray(ordered, dtype=np.int64)
            )
            counts.append(
                np.asarray(
                    [table[k] for k in ordered],
                    dtype=np.int64,
                )
            )

        return LeafReference(
            ids=tuple(ids),
            counts=tuple(counts),
            n_reference=self.n_reference,
        )


@register_scorer
@dataclass(frozen=True)
class InverseLeafOccupancy:
    """Mean inverse reference leaf occupancy."""

    name: ClassVar[str] = "inverse_leaf_occupancy"
    normalization: str = "raw"

    def __post_init__(self):
        if self.normalization not in (
            "raw",
            "reference_percentile",
        ):
            raise ValueError(
                f"Unknown normalization: {self.normalization!r}"
            )

    requirements: ClassVar[Requirements] = Requirements(
        outputs=("leaf_indices",),
        capabilities=("leaf_indices",),
    )

    def prepare(self, box, *, batch_size=1024):
        """Prepare shared reference resources once per fitted Box."""
        from ..models.adapters import get_adapter
        from ..models.ensemble import Ensemble
        from .reference import feature_batches

        models = (
            tuple(box.model.members.values())
            if isinstance(box.model, Ensemble)
            else (box.model,)
        )
        counts_spec = {"type": "leaf_counts", "version": 1}

        def build_counts():
            builders = [LeafReferenceBuilder() for _ in models]
            for X in feature_batches(box, batch_size=batch_size):
                for model, builder in zip(models, builders):
                    leaves = get_adapter(model).leaf_indices(model, X)
                    builder.update(leaves)
            return LeafReferences(builder.finish() for builder in builders)

        references = box.references.get_or_build(
            counts_spec, build_counts
        )
        if self.normalization == "raw":
            return references

        def build_calibration():
            scores = [[] for _ in models]
            for X in feature_batches(box, batch_size=batch_size):
                for i, (model, reference) in enumerate(
                    zip(models, references)
                ):
                    leaves = get_adapter(model).leaf_indices(model, X)
                    scores[i].append(reference.inverse_score(leaves))
            return LeafReferences(
                reference.with_calibration(np.concatenate(parts))
                for reference, parts in zip(references, scores)
            )

        return box.references.get_or_build(
            {
                "type": "leaf_percentile",
                "version": 1,
                "calibration": "empirical_midrank_v1",
            },
            build_calibration
        )

    def __call__(self, context: ScoreContext) -> NDArray:
        leaves = context.resources.get("leaf_indices")
        references = context.state

        if leaves is None or references is None:
            raise ValueError(
                "inverse_leaf_occupancy requires candidate "
                "leaf indices and training reference counts"
            )

        if len(leaves) != len(references):
            raise ValueError(
                "Reference and candidate model counts differ"
            )

        member_scores = np.stack(
            [
                (
                    reference.percentile_score(query)
                    if self.normalization == "reference_percentile"
                    else reference.inverse_score(query)
                )
                for reference, query in zip(
                    references, leaves
                )
            ],
            axis=1,
        )

        weights = context.member_weights

        if weights is None:
            weights = np.ones(len(references))

        weights = np.asarray(weights, dtype=float)

        if (
            weights.shape != (len(references),)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError("Invalid member weights")

        return np.average(
            member_scores,
            axis=1,
            weights=weights,
        )
