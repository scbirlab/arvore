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
        """Reuse cached leaf counts and percentile calibration."""
        from ..models.adapters import get_adapter
        from ..models.ensemble import Ensemble
        from .cache import get_score_cache
        from .reference import feature_batches

        cache = get_score_cache(box)

        models = (
            tuple(box.model.members.values())
            if isinstance(box.model, Ensemble)
            else (box.model,)
        )

        references = cache["leaf_reference"]

        if references is None:
            builders = [
                LeafReferenceBuilder()
                for _ in models
            ]

            for X in feature_batches(
                box, batch_size=batch_size
            ):
                for model, builder in zip(models, builders):
                    leaves = get_adapter(model).leaf_indices(
                        model, X
                    )
                    builder.update(leaves)

            references = tuple(
                builder.finish()
                for builder in builders
            )
            cache["leaf_reference"] = references

        if self.normalization == "raw":
            return references

        calibrated = cache["leaf_percentile"]

        if calibrated is None:
            scores = [[] for _ in models]

            for X in feature_batches(
                box, batch_size=batch_size
            ):
                for i, (model, reference) in enumerate(
                    zip(models, references)
                ):
                    leaves = get_adapter(model).leaf_indices(
                        model, X
                    )
                    scores[i].append(
                        reference.inverse_score(leaves)
                    )

            calibrated = tuple(
                reference.with_calibration(
                    np.concatenate(parts)
                )
                for reference, parts in zip(
                    references, scores
                )
            )

            cache["leaf_percentile"] = calibrated

        return calibrated

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
