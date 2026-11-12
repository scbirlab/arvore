"""Reference-percentile normalisation tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)

from treeton import Box, Ensemble
from treeton.models.adapters import get_adapter
from treeton.scorers import InverseLeafOccupancy
from treeton.scorers.leaves import LeafReferenceBuilder


def test_percentile_midrank_ties():
    leaves = np.array(
        [[0]] * 4 + [[1]] * 2 + [[2]]
    )

    builder = LeafReferenceBuilder()
    builder.update(leaves)
    reference = builder.finish()

    reference = reference.with_calibration(
        reference.inverse_score(leaves)
    )

    queries = np.array([[0], [1], [2], [3]])

    actual = reference.percentile_score(queries)

    expected = np.array([
        2 / 7,
        5 / 7,
        6.5 / 7,
        1.0,
    ])

    np.testing.assert_allclose(actual, expected)


def test_normalization_configuration():
    raw = InverseLeafOccupancy()
    percentile = InverseLeafOccupancy(
        normalization="reference_percentile"
    )

    assert not raw.requirements.reference_score_distribution
    assert percentile.requirements.reference_score_distribution

    with pytest.raises(ValueError, match="Unknown normalization"):
        InverseLeafOccupancy(normalization="invalid")


@pytest.fixture
def example():
    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(60, 4))
    y_train = X_train[:, 0] + X_train[:, 1] ** 2
    X_test = rng.normal(size=(21, 4))

    def dataset(X):
        return Dataset.from_dict({
            f"x{i}": X[:, i]
            for i in range(4)
        })

    train = dataset(X_train)
    test = dataset(X_test)

    members = {
        "rf": RandomForestRegressor(
            n_estimators=11,
            max_depth=3,
            random_state=42,
        ).fit(X_train, y_train),
        "extra": ExtraTreesRegressor(
            n_estimators=17,
            max_depth=5,
            random_state=43,
        ).fit(X_train, y_train),
    }

    box = Box.__new__(Box)
    box.model = Ensemble(members, weights=(1, 3))
    box.features = tuple(f"x{i}" for i in range(4))
    box.target = ("y",)

    pipeline = SimpleNamespace(
        data_in=None,
        data_out=train,
        column_transforms={},
    )

    box.pipeline = pipeline
    box._pre_predict = lambda **kwargs: (test, pipeline)

    return box, X_train, X_test, members


def test_ensemble_reference_percentile(example):
    box, X_train, X_test, members = example

    scorer = InverseLeafOccupancy(
        normalization="reference_percentile"
    )

    result = box.predict(
        score=scorer,
        disaggregate=True,
        batch_size=8,
        map_kwargs={"load_from_cache_file": False},
    )

    per_member = []

    for model in members.values():
        adapter = get_adapter(model)

        train_leaves = adapter.leaf_indices(model, X_train)
        test_leaves = adapter.leaf_indices(model, X_test)

        builder = LeafReferenceBuilder()
        builder.update(train_leaves)

        reference = builder.finish().with_calibration(
            builder.finish().inverse_score(train_leaves)
        )

        per_member.append(
            reference.percentile_score(test_leaves)
        )

    expected = np.average(
        np.stack(per_member, axis=1),
        axis=1,
        weights=box.model.weights,
    )

    np.testing.assert_allclose(
        result["score:inverse_leaf_occupancy"],
        expected,
    )

    scores = np.asarray(
        result["score:inverse_leaf_occupancy"]
    )
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)

    for name, member in members.items():
        np.testing.assert_allclose(
            result[f"prediction:y:{name}"],
            member.predict(X_test),
        )


def test_percentile_batch_independence(example):
    box, _, _, _ = example

    scorer = InverseLeafOccupancy(
        normalization="reference_percentile"
    )

    results = [
        box.predict(
            score=scorer,
            batch_size=size,
            map_kwargs={"load_from_cache_file": False},
        )
        for size in (3, 21)
    ]

    np.testing.assert_allclose(
        results[0]["score:inverse_leaf_occupancy"],
        results[1]["score:inverse_leaf_occupancy"],
    )


def test_raw_default_unchanged(example):
    box, _, _, _ = example

    by_name = box.predict(
        score="inverse_leaf_occupancy",
        map_kwargs={"load_from_cache_file": False},
    )

    explicit = box.predict(
        score=InverseLeafOccupancy(normalization="raw"),
        map_kwargs={"load_from_cache_file": False},
    )

    np.testing.assert_allclose(
        by_name["score:inverse_leaf_occupancy"],
        explicit["score:inverse_leaf_occupancy"],
    )
