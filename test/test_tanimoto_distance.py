"""Tests for exact Tanimoto reference distance."""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor

from treeton import Box, Ensemble
from treeton.scorers import TanimotoDistance
from treeton.scorers.tanimoto import TanimotoReference


def brute_force_distance(queries, references):
    """Independent binary Tanimoto implementation."""
    q = np.asarray(queries, dtype=bool)
    r = np.asarray(references, dtype=bool)

    result = []

    for query in q:
        similarities = []

        for reference in r:
            intersection = np.logical_and(
                query, reference
            ).sum()
            union = np.logical_or(
                query, reference
            ).sum()

            similarity = (
                intersection / union if union else 1.0
            )
            similarities.append(similarity)

        result.append(1.0 - max(similarities))

    return np.asarray(result)


def test_tanimoto_exact_and_chunked():
    rng = np.random.default_rng(42)

    references = rng.integers(
        0, 2, size=(37, 129), dtype=np.uint8
    )
    queries = rng.integers(
        0, 2, size=(13, 129), dtype=np.uint8
    )

    expected = brute_force_distance(
        queries, references
    )
    index = TanimotoReference.build(references)

    for q_chunk, r_chunk in [(1, 1), (3, 7), (32, 512)]:
        np.testing.assert_allclose(
            index.distance(
                queries,
                query_chunk=q_chunk,
                reference_chunk=r_chunk,
            ),
            expected,
            atol=1e-12,
        )


def test_tanimoto_zero_vectors():
    index = TanimotoReference.build([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ])

    np.testing.assert_allclose(
        index.distance([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]),
        [0.0, 1.0, 0.0],
    )


def test_invalid_fingerprints():
    with pytest.raises(ValueError, match="only 0 and 1"):
        TanimotoReference.build([[0, 2, 1]])

    index = TanimotoReference.build([[0, 1, 1]])

    with pytest.raises(ValueError, match="Expected 3"):
        index.distance([[0, 1, 1, 0]])


@pytest.fixture
def example():
    rng = np.random.default_rng(5)

    X_train = rng.normal(size=(30, 2))
    X_test = rng.normal(size=(12, 2))

    y = X_train[:, 0] + X_train[:, 1] ** 2

    train_fp = rng.integers(
        0, 2, size=(30, 64)
    ).tolist()
    test_fp = rng.integers(
        0, 2, size=(12, 64)
    ).tolist()

    def make_dataset(X, fp):
        return Dataset.from_dict({
            "x0": X[:, 0],
            "x1": X[:, 1],
            "fingerprint": fp,
        })

    train = make_dataset(X_train, train_fp)
    test = make_dataset(X_test, test_fp)

    model = RandomForestRegressor(
        n_estimators=11,
        random_state=42,
    ).fit(X_train, y)

    box = Box.__new__(Box)
    box.model = model
    box.features = ("x0", "x1")
    box.target = ("y",)

    pipeline = SimpleNamespace(
        data_in=None,
        data_out=train,
        column_transforms={},
    )

    box.pipeline = pipeline
    box._pre_predict = lambda **kwargs: (test, pipeline)

    return box, X_test, train_fp, test_fp


def test_tanimoto_box(example):
    box, X_test, train_fp, test_fp = example

    result = box.predict(
        score=[
            "tanimoto_distance",
            "euclidean_distance",
        ],
        batch_size=5,
        map_kwargs={"load_from_cache_file": False},
    )

    expected = brute_force_distance(
        test_fp, train_fp
    )

    np.testing.assert_allclose(
        result["score:tanimoto_distance"],
        expected,
    )

    assert "score:euclidean_distance" in result.column_names

    np.testing.assert_allclose(
        result["prediction:y"],
        box.model.predict(X_test),
    )


def test_disaggregate_preserved(example):
    from copy import deepcopy

    box, _, _, _ = example
    original = box.model

    box.model = Ensemble({
        "a": original,
        "b": deepcopy(original),
    })

    baseline = box.predict(
        disaggregate=True,
        map_kwargs={"load_from_cache_file": False},
    )

    scored = box.predict(
        disaggregate=True,
        score="tanimoto_distance",
        map_kwargs={"load_from_cache_file": False},
    )

    for name in ("a", "b"):
        column = f"prediction:y:{name}"

        np.testing.assert_allclose(
            baseline[column],
            scored[column],
        )

    assert "score:tanimoto_distance" in scored.column_names


def test_fingerprint_generation_hook(monkeypatch):
    """Exercise SMILES fallback without requiring chemistry extras."""

    alphabet = "CNOS"

    def fingerprint(smiles):
        return [int(atom in smiles) for atom in alphabet]

    class FakeFingerprintPipeline:
        def __call__(self, data):
            return SimpleNamespace(
                data=Dataset.from_dict({
                    "fingerprint": [
                        fingerprint(s)
                        for s in data["smiles"]
                    ],
                })
            )

    monkeypatch.setattr(
        TanimotoDistance,
        "_fingerprint_pipeline",
        lambda self: FakeFingerprintPipeline(),
    )

    scorer = TanimotoDistance(
        reference_batch_size=2,
    )

    reference = Dataset.from_dict({
        "smiles": ["CC", "CN", "OO", "SS"],
    })

    candidates = Dataset.from_dict({
        "smiles": ["CC", "NO", "NN"],
    })

    prepared = scorer.prepare_candidates(candidates)
    index = scorer.prepare_reference(reference)

    expected = brute_force_distance(
        [fingerprint(s) for s in candidates["smiles"]],
        [fingerprint(s) for s in reference["smiles"]],
    )

    np.testing.assert_allclose(
        index.distance(prepared["fingerprint"]),
        expected,
    )

    assert "fingerprint" not in candidates.column_names
    assert "fingerprint" in prepared.column_names


def test_missing_fingerprint_and_structure():
    scorer = TanimotoDistance()

    data = Dataset.from_dict({
        "x": [1, 2, 3],
    })

    with pytest.raises(ValueError, match="requires fingerprint"):
        scorer.prepare_candidates(data)
