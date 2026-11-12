"""Exact nearest-reference Tanimoto distance for binary fingerprints."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import Requirements, ScoreContext
from .registry import register_scorer


# Compatible with NumPy 1.26; no np.bit_count dependency.
_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1).astype(np.uint8)


def _encode_fingerprints(fingerprints, n_bits=None):
    """Validate and bit-pack a matrix of binary fingerprints."""
    bits = np.asarray(fingerprints)

    if bits.ndim != 2:
        raise ValueError(
            "Fingerprints must have shape (samples, bits)"
        )

    if bits.shape[1] == 0:
        raise ValueError("Fingerprint width must be positive")

    if n_bits is not None and bits.shape[1] != n_bits:
        raise ValueError(
            f"Expected {n_bits} fingerprint bits, "
            f"received {bits.shape[1]}"
        )

    if not (
        np.issubdtype(bits.dtype, np.number)
        or np.issubdtype(bits.dtype, np.bool_)
    ):
        raise TypeError("Fingerprints must be numeric or boolean")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError(
            "Tanimoto fingerprints must contain only 0 and 1"
        )

    packed = np.packbits(
        bits.astype(np.uint8, copy=False),
        axis=1,
        bitorder="little",
    )

    counts = _POPCOUNT[packed].sum(
        axis=1,
        dtype=np.int32,
    )

    return packed, counts, bits.shape[1]


@dataclass(frozen=True)
class TanimotoReference:
    """Packed binary fingerprints and precomputed bit counts."""

    packed: NDArray
    counts: NDArray
    n_bits: int

    @classmethod
    def build(cls, fingerprints):
        packed, counts, n_bits = _encode_fingerprints(
            fingerprints
        )

        if not len(packed):
            raise ValueError("Reference fingerprints are empty")

        return cls(packed, counts, n_bits)

    def distance(
        self,
        fingerprints,
        *,
        query_chunk: int = 32,
        reference_chunk: int = 512,
    ) -> NDArray:
        """Exact 1 - maximum Tanimoto similarity."""
        if query_chunk < 1 or reference_chunk < 1:
            raise ValueError("Chunk sizes must be positive")

        queries, q_counts, _ = _encode_fingerprints(
            fingerprints,
            n_bits=self.n_bits,
        )

        best = np.zeros(len(queries), dtype=np.float64)

        for q_start in range(0, len(queries), query_chunk):
            q_end = q_start + query_chunk
            q = queries[q_start:q_end]
            q_count = q_counts[q_start:q_end]

            q_best = np.zeros(len(q), dtype=np.float64)

            for r_start in range(
                0, len(self.packed), reference_chunk
            ):
                r_end = r_start + reference_chunk
                r = self.packed[r_start:r_end]
                r_count = self.counts[r_start:r_end]

                intersection = _POPCOUNT[
                    np.bitwise_and(
                        q[:, None, :],
                        r[None, :, :],
                    )
                ].sum(axis=2, dtype=np.int32)

                union = (
                    q_count[:, None]
                    + r_count[None, :]
                    - intersection
                )

                # Both-empty fingerprints have similarity 1.
                similarity = np.divide(
                    intersection,
                    union,
                    out=np.ones(
                        union.shape,
                        dtype=np.float64,
                    ),
                    where=union != 0,
                )

                q_best = np.maximum(
                    q_best,
                    similarity.max(axis=1),
                )

            best[q_start:q_end] = q_best

        return np.clip(1.0 - best, 0.0, 1.0)


@register_scorer
@dataclass(frozen=True)
class TanimotoDistance:
    """Nearest-reference Tanimoto distance."""

    name: ClassVar[str] = "tanimoto_distance"

    column: str = "fingerprint"
    structure_column: str = "smiles"
    fingerprint_transform: str = "morgan-fingerprint"
    reference_batch_size: int = 2048
    query_chunk: int = 32
    reference_chunk: int = 512

    def __post_init__(self):
        if min(
            self.reference_batch_size,
            self.query_chunk,
            self.reference_chunk,
        ) < 1:
            raise ValueError("Batch and chunk sizes must be positive")

    @property
    def requirements(self) -> Requirements:
        return Requirements(
            columns=(self.column,),
            reference_columns=(self.column,),
            reference_indices=("tanimoto",),
        )

    def _fingerprint_pipeline(self):
        from aspect import DataPipeline

        return DataPipeline({
            self.column: (
                self.structure_column,
                self.fingerprint_transform,
            ),
        })

    def _add_fingerprint(self, data):
        """Use existing fingerprints or derive them from structures."""
        if self.column in data.column_names:
            return data

        if self.structure_column not in data.column_names:
            raise ValueError(
                f"Tanimoto requires fingerprint column "
                f"{self.column!r} or structure column "
                f"{self.structure_column!r}; available: "
                f"{data.column_names}"
            )

        source = data.select_columns(
            [self.structure_column]
        )

        transformed = self._fingerprint_pipeline()(source)

        # Aspect returns a Providence DataResolver.
        if hasattr(transformed, "data"):
            transformed = transformed.data

        fingerprints = transformed.with_format(None)[
            self.column
        ]

        return data.add_column(
            self.column,
            fingerprints,
        )

    def prepare_candidates(self, data):
        """Prepare auxiliary columns without changing model features."""
        return self._add_fingerprint(data)

    def prepare_reference(self, data) -> TanimotoReference:
        """Build the packed reference index in bounded batches."""
        from datasets import Dataset

        if not isinstance(data, Dataset):
            if isinstance(data, dict):
                data = Dataset.from_dict(data)
            else:
                raise TypeError(
                    "Tanimoto reference requires a Dataset "
                    "or a mapping of columns"
                )

        packed_parts = []
        count_parts = []
        n_bits = None

        for start in range(
            0, len(data), self.reference_batch_size
        ):
            batch = data.select(
                range(
                    start,
                    min(
                        start + self.reference_batch_size,
                        len(data),
                    ),
                )
            )

            batch = self._add_fingerprint(batch)

            packed, counts, width = _encode_fingerprints(
                batch.with_format(None)[self.column],
                n_bits=n_bits,
            )

            if n_bits is None:
                n_bits = width

            packed_parts.append(packed)
            count_parts.append(counts)

        if not packed_parts:
            raise ValueError("Training reference is empty")

        return TanimotoReference(
            packed=np.concatenate(packed_parts, axis=0),
            counts=np.concatenate(count_parts, axis=0),
            n_bits=n_bits,
        )

    def __call__(self, context: ScoreContext) -> NDArray:
        reference = context.resources.get(
            "tanimoto_reference"
        )

        if reference is None:
            raise ValueError(
                "Tanimoto reference has not been prepared"
            )

        if self.column not in context.batch:
            raise ValueError(
                f"Candidate fingerprint column "
                f"{self.column!r} is missing"
            )

        return reference.distance(
            context.batch[self.column],
            query_chunk=self.query_chunk,
            reference_chunk=self.reference_chunk,
        )
