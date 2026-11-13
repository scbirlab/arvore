"""Euclidean reference-distance scoring tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor

from treeton import Box, Ensemble
from treeton.scorers import FeatureDistance
from treeton.scorers.distances import FeatureReference


@pytest.fixture
def example():
    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(50, 4))
    X_test = rng.normal(size=(19, 4))
    y = X_train[:, 0] - X_train[:, 1] ** 2

    model = RandomForestRegressor(
        n_estimators=15,
        random_state=42,
    ).fit(X_train, y)

    def dataset(X):
        return Dataset.from_dict({
            f"x{i}": X[:, i]
            for i in range(4)
        })

    train = dataset(X_train)
    test = dataset(X_test)

    box = Box.__new__(Box)
    box.model = model
    box.features = tuple(f"x{i}" for i in range(4))
    box.target = ("y",)

    pipeline = SimpleNamespace(
        data_in=None,
        data_out=train,
        column_transforms={},
    )

    box.pipeline = pipeline
    box._pre_predict = lambda **kwargs: (test, pipeline)

    return box, X_train, X_test, model


def test_reference_matches_brute_force():
    rng = np.random.default_rng(1)

    X_train = rng.normal(size=(35, 5))
    X_test = rng.normal(size=(11, 5))

    reference = FeatureReference.build(X_train)

    expected = np.linalg.norm(
        X_test[:, None, :] - X_train[None, :, :],
        axis=2,
    ).min(axis=1)

    np.testing.assert_allclose(
        reference.distance(X_test),
        expected,
        atol=1e-7,
    )


def test_euclidean_box_scoring(example):
    box, X_train, X_test, model = example

    result = box.predict(
        score="feature_distance",
        batch_size=6,
        map_kwargs={"load_from_cache_file": False},
    )

    expected = np.linalg.norm(
        X_test[:, None, :] - X_train[None, :, :],
        axis=2,
    ).min(axis=1)

    np.testing.assert_allclose(
        result["score:feature_distance"],
        expected,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        result["prediction:y"],
        model.predict(X_test),
    )


def test_training_rows_have_zero_distance(example):
    box, X_train, _, _ = example

    index = FeatureDistance().prepare(box, batch_size=7)

    np.testing.assert_allclose(
        index.distance(X_train),
        np.zeros(len(X_train)),
        atol=1e-7,
    )


def test_euclidean_batch_independence(example):
    box, _, _, _ = example

    results = [
        box.predict(
            score=FeatureDistance(),
            batch_size=batch_size,
            map_kwargs={"load_from_cache_file": False},
        )
        for batch_size in (3, 19)
    ]

    np.testing.assert_allclose(
        results[0]["score:feature_distance"],
        results[1]["score:feature_distance"],
    )


def test_euclidean_and_leaf_scoring(example):
    from copy import deepcopy

    box, _, _, model = example

    box.model = Ensemble({
        "first": model,
        "second": deepcopy(model),
    })

    result = box.predict(
        score=[
            "feature_distance",
            "inverse_leaf_occupancy",
            "ensemble_sd",
            "tree_sd",
        ],
        disaggregate=True,
        batch_size=7,
        map_kwargs={"load_from_cache_file": False},
    )

    assert "score:feature_distance" in result.column_names
    assert "score:inverse_leaf_occupancy" in result.column_names
    assert "score:ensemble_sd:y" in result.column_names
    assert "score:tree_sd:y" in result.column_names

    for name in box.model.members:
        assert f"prediction:y:{name}" in result.column_names


def test_invalid_reference_features():
    with pytest.raises(ValueError, match="finite"):
        FeatureReference.build(
            [[0.0, float("nan")]]
        )


def test_invalid_candidate_dimension():
    reference = FeatureReference.build(
        np.zeros((3, 4))
    )

    with pytest.raises(ValueError, match="Expected"):
        reference.distance(
            np.zeros((2, 5))
        )
