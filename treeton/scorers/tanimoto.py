"""Exact nearest-reference Tanimoto distance for binary fingerprints."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .base import Requirements, ScoreContext
from .registry import register_scorer


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

    counts = np.bitwise_count(packed).sum(
        axis=1,
        dtype=np.int32,
    )
    return packed, counts, bits.shape[1]


@dataclass(frozen=True)
class TanimotoReference:
    """Packed fingerprints for exact Tanimoto nearest-neighbour search."""

    packed: NDArray
    counts: NDArray
    n_bits: int

    @classmethod
    def build(cls, fingerprints):
        packed, counts, n_bits = _encode_fingerprints(fingerprints)
        if not len(packed):
            raise ValueError("Reference fingerprints are empty")
        return cls(packed, counts, n_bits)

    def state_dict(self):
        return {
            "packed": np.asarray(self.packed, dtype=np.uint8),
            "counts": np.asarray(self.counts, dtype=np.int32),
            "n_bits": np.array(self.n_bits, dtype=np.int64),
        }

    @classmethod
    def from_state_dict(cls, state, spec):
        packed = np.asarray(state["packed"])
        counts = np.asarray(state["counts"])
        width = np.asarray(state["n_bits"])

        if width.shape != () or not np.issubdtype(
            width.dtype, np.integer
        ):
            raise ValueError("Invalid fingerprint width")

        n_bits = int(width)
        if (
            packed.ndim != 2
            or len(packed) == 0
            or packed.dtype != np.uint8
            or counts.ndim != 1
            or counts.dtype != np.int32
            or len(counts) != len(packed)
            or n_bits < 1
            or packed.shape[1] != (n_bits + 7) // 8
            or np.any(
                counts != np.bitwise_count(packed).sum(
                    axis=1, dtype=np.int32
                )
            )
        ):
            raise ValueError("Invalid Tanimoto reference archive")

        if n_bits % 8:
            mask = 255 ^ ((1 << (n_bits % 8)) - 1)
            if np.any(packed[:, -1] & mask):
                raise ValueError("Nonzero fingerprint padding bits")

        return cls(packed.copy(), counts.copy(), n_bits)

    def distance(
        self,
        fingerprints,
        *,
        n_neighbors=1,
        return_indices=False,
        query_chunk=32,
        reference_chunk=512,
    ):
        """Mean exact distance to k closest references.

        Indices, when requested, are ordered by increasing distance,
        breaking similarity ties by original training-row index.
        """
        if not isinstance(n_neighbors, int) or not (
            1 <= n_neighbors <= len(self.packed)
        ):
            raise ValueError("Invalid number of neighbours")
        if query_chunk < 1 or reference_chunk < 1:
            raise ValueError("Chunk sizes must be positive")

        queries, q_counts, _ = _encode_fingerprints(
            fingerprints, n_bits=self.n_bits
        )
        distances = np.empty(len(queries), dtype=np.float64)
        neighbors = (
            np.empty((len(queries), n_neighbors), dtype=np.int64)
            if return_indices else None
        )

        for start in range(0, len(queries), query_chunk):
            q = queries[start:start + query_chunk]
            qc = q_counts[start:start + query_chunk]
            width = len(q)

            best_sim = np.full(
                (width, n_neighbors), -np.inf, dtype=np.float64
            )
            best_idx = np.full(
                (width, n_neighbors),
                len(self.packed),
                dtype=np.int64,
            )

            for offset in range(0, len(self.packed), reference_chunk):
                r = self.packed[offset:offset + reference_chunk]
                rc = self.counts[offset:offset + reference_chunk]

                intersection = np.bitwise_count(
                    np.bitwise_and(q[:, None, :], r[None, :, :])
                ).sum(axis=2, dtype=np.int32)

                union = qc[:, None] + rc[None, :] - intersection
                similarity = np.divide(
                    intersection,
                    union,
                    out=np.ones(union.shape, dtype=np.float64),
                    where=union != 0,
                )

                index = np.broadcast_to(
                    np.arange(offset, offset + len(r), dtype=np.int64),
                    similarity.shape,
                )

                candidate_sim = np.concatenate(
                    (best_sim, similarity), axis=1
                )
                candidate_idx = np.concatenate(
                    (best_idx, index), axis=1
                )

                # Deterministic tie-breaking, vectorised over queries.
                order = np.lexsort(
                    (candidate_idx, -candidate_sim), axis=1
                )[:, :n_neighbors]

                best_sim = np.take_along_axis(
                    candidate_sim, order, axis=1
                )
                best_idx = np.take_along_axis(
                    candidate_idx, order, axis=1
                )

            distances[start:start + width] = (
                1.0 - best_sim.mean(axis=1)
            )
            if neighbors is not None:
                neighbors[start:start + width] = best_idx

        distances = np.clip(distances, 0.0, 1.0)
        if return_indices:
            return distances, neighbors
        return distances


@dataclass(frozen=True)
class ReferenceLabels:
    """Row-aligned string representations of reference observations."""

    values: NDArray

    def state_dict(self):
        return {"values": self.values}

    @classmethod
    def from_state_dict(cls, state, spec):
        values = np.asarray(state["values"])
        if values.ndim != 1 or values.dtype.kind != "U":
            raise ValueError("Invalid reference labels")
        return cls(values.copy())


@register_scorer
@dataclass(frozen=True)
class TanimotoDistance:
    """Nearest-reference Tanimoto distance."""

    name: ClassVar[str] = "tanimoto_distance"
    version: ClassVar[int] = 1
    
    column: str = "fingerprint"
    structure_column: str = "smiles"
    fingerprint_transform: str = "morgan-fingerprint"
    radius: int = 2
    n_neighbors: int = 1
    reference_batch_size: int = 2048
    query_chunk: int = 32
    reference_chunk: int = 512
    return_neighbors: bool = False
    neighbor_column: str = "smiles"
    neighbor_delimiter: str = ";"

    def __post_init__(self):
        if not isinstance(self.radius, int) or self.radius < 0:
            raise ValueError("radius must be nonnegative")
        if not isinstance(self.neighbor_column, str) or not self.neighbor_column:
            raise ValueError("neighbor_column must be a nonempty string")
        if not isinstance(self.neighbor_delimiter, str):
            raise ValueError("neighbor_delimiter must be a string")
        if min(
            self.n_neighbors,
            self.reference_batch_size,
            self.query_chunk,
            self.reference_chunk,
        ) < 1:
            raise ValueError("Batch and chunk sizes must be positive")

    @property
    def requirements(self) -> Requirements:
        return Requirements(
            columns=(self.column,),
        )

    def _fingerprint_pipeline(self):
        from aspect import DataPipeline

        return DataPipeline({
            self.column: (
                self.structure_column,
                {
                    "name": self.fingerprint_transform,
                    "kwargs": {
                        "radius": self.radius,
                    },
                },
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

    def prepare_candidates(self, data, *, box):
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

    def _reference_data(self, box):
        """Find raw or retained rows in reference order."""
        data = box._training_reference()
        required = {self.column}
        if self.return_neighbors:
            required.add(self.neighbor_column)

        if not required.issubset(set(data.column_names)):
            source = getattr(box.pipeline, "data_in", None)
            if source is not None:
                source = box._unwrap(source)
                if hasattr(source, "with_format"):
                    source = source.with_format(None)
                if hasattr(source, "column_names"):
                    if (
                        self.column in source.column_names
                        or self.structure_column in source.column_names
                    ):
                        data = source
        return data

    def prepare(self, box, *, batch_size=1024):
        """Cache packed fingerprints and optional aligned labels separately."""
        spec = {
            "type": "tanimoto",
            "version": self.version,
            "column": self.column,
            "structure_column": self.structure_column,
            "fingerprint_transform": self.fingerprint_transform,
            "radius": self.radius,
            "encoding": "packbits-little",
        }

        reference = box.references.get_or_build(
            spec,
            lambda: self.prepare_reference(self._reference_data(box)),
        )

        if not self.return_neighbors:
            return reference

        label_spec = {
            "type": "tanimoto_labels",
            "version": 1,
            "column": self.neighbor_column,
            "fingerprint_reference": spec,
        }

        def build_labels():
            data = self._reference_data(box)
            if self.neighbor_column not in data.column_names:
                raise ValueError(
                    f"Training reference lacks neighbour column "
                    f"{self.neighbor_column!r}"
                )
            values = data.with_format(None)[self.neighbor_column]
            if any(not isinstance(value, str) for value in values):
                raise ValueError(
                    "Neighbour representations must be strings"
                )
            if len(values) != len(reference.packed):
                raise ValueError(
                    "Neighbour labels and fingerprints are not row-aligned"
                )
            return ReferenceLabels(np.asarray(values, dtype=str))

        labels = box.references.get_or_build(label_spec, build_labels)
        if len(labels.values) != len(reference.packed):
            raise ValueError("Mismatched cached neighbour labels")
        return reference, labels

    def __call__(self, context: ScoreContext):
        if self.column not in context.batch:
            raise ValueError(
                f"Candidate fingerprint column {self.column!r} is missing"
            )

        if self.return_neighbors:
            reference, labels = context.state
            distances, indices = reference.distance(
                context.batch[self.column],
                n_neighbors=self.n_neighbors,
                return_indices=True,
                query_chunk=self.query_chunk,
                reference_chunk=self.reference_chunk,
            )
            representations = labels.values[indices]
            return {
                "": distances,
                "neighbor": [
                    self.neighbor_delimiter.join(row)
                    for row in representations
                ],
            }

        return context.state.distance(
            context.batch[self.column],
            n_neighbors=self.n_neighbors,
            query_chunk=self.query_chunk,
            reference_chunk=self.reference_chunk,
        )
