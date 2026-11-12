"""Regression tests for scorer-owned preparation and execution."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor

from treeton import Box, Ensemble
from treeton.scorers import (
    EuclideanDistance,
    InverseLeafOccupancy,
    TanimotoDistance,
)
from treeton.scorers.base import Requirements


@pytest.fixture
def example():
    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(40, 3))
    X_test = rng.normal(size=(13, 3))
    y = X_train[:, 0] + X_train[:, 1] ** 2

    train_fp = rng.integers(
        0, 2, size=(40, 32)
    ).tolist()

    test_fp = rng.integers(
        0, 2, size=(13, 32)
    ).tolist()

    def dataset(X, fingerprints):
        return Dataset.from_dict({
            "x0": X[:, 0],
            "x1": X[:, 1],
            "x2": X[:, 2],
            "fingerprint": fingerprints,
            "other_fp": fingerprints,
        })

    train = dataset(X_train, train_fp)
    test = dataset(X_test, test_fp)

    model = RandomForestRegressor(
        n_estimators=11,
        random_state=42,
    ).fit(X_train, y)

    box = Box.__new__(Box)
    box.model = model
    box.features = ("x0", "x1", "x2")
    box.target = ("y",)

    pipeline = SimpleNamespace(
        data_in=None,
        data_out=train,
        column_transforms={},
    )

    box.pipeline = pipeline
    box._pre_predict = lambda **kwargs: (test, pipeline)

    return box, X_test


def predict(box, **kwargs):
    return box.predict(
        batch_size=4,
        map_kwargs={"load_from_cache_file": False},
        **kwargs,
    )


def test_custom_scorer_lifecycle(example):
    box, X = example

    class CustomScore:
        name = "custom_score"

        def __init__(self):
            self.prepares = 0
            self.candidate_calls = 0

        @property
        def requirements(self):
            return Requirements(columns=("aux",))

        def prepare_candidates(self, data, *, box):
            assert box is not None
            self.candidate_calls += 1
            return data.add_column(
                "aux", [5.0] * len(data)
            )

        def validate(self, box, prepared):
            assert "aux" in prepared.column_names
            assert box.model is not None

        def prepare(self, box, *, batch_size):
            self.prepares += 1
            assert batch_size == 4
            return {"offset": 2.0}

        def __call__(self, context):
            assert context.box is box
            assert context.state == {"offset": 2.0}
            return (
                np.asarray(context.batch["aux"])
                + context.state["offset"]
            )

    scorer = CustomScore()
    result = predict(box, score=scorer)

    assert scorer.prepares == 1
    assert scorer.candidate_calls == 1

    np.testing.assert_allclose(
        result["score:custom_score"],
        np.full(len(X), 7.0),
    )


def test_scoring_does_not_change_predictions(example):
    box, X = example
    box.model = Ensemble({
        "a": box.model,
        "b": deepcopy(box.model),
    })

    for disaggregate in (False, True):
        baseline = predict(
            box, disaggregate=disaggregate
        )

        scored = predict(
            box,
            disaggregate=disaggregate,
            score=[
                "ensemble_sd",
                "tree_sd",
                "inverse_leaf_occupancy",
                "euclidean_distance",
                "tanimoto_distance",
            ],
        )

        columns = [
            column
            for column in baseline.column_names
            if column.startswith("prediction:")
        ]

        assert columns == [
            column
            for column in scored.column_names
            if column.startswith("prediction:")
        ]

        for column in columns:
            np.testing.assert_allclose(
                baseline[column],
                scored[column],
            )

        assert "score:euclidean_distance" in scored.column_names
        assert "score:tanimoto_distance" in scored.column_names
        assert (
            "score:inverse_leaf_occupancy"
            in scored.column_names
        )
        assert "score:tree_sd:y" in scored.column_names
        assert "score:ensemble_sd:y" in scored.column_names


def test_custom_callable_sees_box(example):
    box, X = example

    def inspect_box(context):
        assert context.box is box
        assert context.state is None
        return context.X[:, 0]

    result = predict(
        box, score=inspect_box
    )

    np.testing.assert_allclose(
        result["score:inspect_box"],
        X[:, 0],
    )


def test_explicit_score_aliases(example):
    box, _ = example

    result = predict(
        box,
        score={
            "fingerprint_a": TanimotoDistance(
                column="fingerprint"
            ),
            "fingerprint_b": TanimotoDistance(
                column="other_fp"
            ),
        },
    )

    assert "score:fingerprint_a" in result.column_names
    assert "score:fingerprint_b" in result.column_names

    np.testing.assert_allclose(
        result["score:fingerprint_a"],
        result["score:fingerprint_b"],
    )


def test_reused_scorer_has_independent_state(example):
    box, _ = example
    scorer = InverseLeafOccupancy(
        normalization="reference_percentile"
    )

    first = predict(box, score=scorer)
    second = predict(box, score=scorer)

    np.testing.assert_allclose(
        first["score:inverse_leaf_occupancy"],
        second["score:inverse_leaf_occupancy"],
    )


def test_reference_scorers_prepare_directly(example):
    box, X = example

    euclidean = EuclideanDistance().prepare(
        box, batch_size=7
    )

    tanimoto = TanimotoDistance().prepare(
        box, batch_size=7
    )

    assert euclidean.n_reference == 40
    assert len(tanimoto.packed) == 40
    assert len(euclidean.distance(X)) == len(X)
