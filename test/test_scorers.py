"""Tests for prediction scorer interfaces."""

import numpy as np
import pytest

from treeton.scorers import (
    EnsembleSD,
    Requirements,
    ScoreContext,
    resolve_scorers,
)


def make_context(weights=(1.0, 3.0)):
    predictions = np.array([
        [[1.0, 5.0], [3.0, 9.0]],
        [[2.0, 4.0], [6.0, 8.0]],
    ])

    return ScoreContext(
        X=np.zeros((2, 3)),
        predictions=np.average(
            predictions,
            axis=1,
            weights=weights,
        ),
        model=None,
        batch={},
        targets=("target_a", "target_b"),
        member_predictions=predictions,
        member_weights=np.asarray(weights),
    )


def test_ensemble_sd():
    context = make_context()

    result = EnsembleSD()(context)

    expected = np.array([
        [np.sqrt(0.75), np.sqrt(3.0)],
        [np.sqrt(3.0), np.sqrt(3.0)],
    ])

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected)


def test_ensemble_sd_equal_members():
    context = make_context(weights=(1.0, 1.0))

    result = EnsembleSD()(context)

    np.testing.assert_allclose(
        result,
        [[1.0, 2.0], [2.0, 2.0]],
    )


def test_ensemble_sd_requires_multiple_members():
    context = make_context()

    context = ScoreContext(
        X=context.X,
        predictions=context.predictions,
        model=None,
        batch={},
        targets=context.targets,
        member_predictions=context.member_predictions[:, :1],
        member_weights=np.ones(1),
    )

    with pytest.raises(ValueError, match="at least two"):
        EnsembleSD()(context)


def test_scorer_resolution():
    scorer, = resolve_scorers("ensemble_sd")

    assert isinstance(scorer, EnsembleSD)
    assert scorer.requirements == Requirements(
        outputs=("member_predictions",),
        capabilities=("ensemble_members",),
    )


def test_custom_callable():
    def constant_score(context):
        return np.ones(len(context.X))

    resolved = resolve_scorers(constant_score)

    assert resolved == (constant_score,)


def test_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_scorers([
            "ensemble_sd",
            EnsembleSD(),
        ])


# Box integration tests

from types import SimpleNamespace

from datasets import Dataset

from treeton import Box, Ensemble


class CountingEstimator:
    """Deterministic predictor tracking model calls."""

    def __init__(self, values):
        self.values = np.atleast_1d(
            np.asarray(values, dtype=float)
        )
        self.calls = 0

    def fit(self, X, y):
        return self

    def predict(self, X, **kwargs):
        self.calls += 1
        return np.tile(self.values, (len(X), 1))


def make_prediction_box(
    values=((1.0,), (3.0,)),
    weights=(1.0, 3.0),
    targets=("y",),
):
    members = {
        f"member_{i}": CountingEstimator(value)
        for i, value in enumerate(values)
    }

    box = Box.__new__(Box)
    box.model = Ensemble(members, weights=weights)
    box.features = ("x",)
    box.target = tuple(targets)

    data = Dataset.from_dict({
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
    })

    # Isolate Box.predict from Aspect preprocessing.
    box._pre_predict = lambda **kwargs: (
        data,
        SimpleNamespace(column_transforms={}),
    )

    return box, members


def test_box_predict_ensemble_sd():
    box, members = make_prediction_box()

    result = box.predict(
        score="ensemble_sd",
        batch_size=2,
        map_kwargs={"load_from_cache_file": False},
    )

    assert "prediction:y" in result.column_names
    assert "score:ensemble_sd:y" in result.column_names

    np.testing.assert_allclose(
        result["prediction:y"],
        np.full(5, 2.5),
    )

    np.testing.assert_allclose(
        result["score:ensemble_sd:y"],
        np.full(5, np.sqrt(0.75)),
    )

    # Five rows, three batches.
    # Each member is called exactly once per batch.
    assert all(m.calls == 3 for m in members.values())


def test_box_predict_multitarget():
    box, _ = make_prediction_box(
        values=((1.0, 10.0), (3.0, 14.0)),
        weights=(1.0, 1.0),
        targets=("a", "b"),
    )

    result = box.predict(
        score="ensemble_sd",
        batch_size=3,
        map_kwargs={"load_from_cache_file": False},
    )

    np.testing.assert_allclose(
        result["prediction:a"],
        np.full(5, 2.0),
    )

    np.testing.assert_allclose(
        result["prediction:b"],
        np.full(5, 12.0),
    )

    np.testing.assert_allclose(
        result["score:ensemble_sd:a"],
        np.full(5, 1.0),
    )

    np.testing.assert_allclose(
        result["score:ensemble_sd:b"],
        np.full(5, 2.0),
    )


def test_box_predict_custom_callable():
    box, _ = make_prediction_box()

    def feature_norm(context):
        return np.linalg.norm(context.X, axis=1)

    result = box.predict(
        score=feature_norm,
        batch_size=2,
        map_kwargs={"load_from_cache_file": False},
    )

    assert "score:feature_norm" in result.column_names

    np.testing.assert_allclose(
        result["score:feature_norm"],
        np.arange(5, dtype=float),
    )


def test_box_predict_invalid_score_shape():
    box, _ = make_prediction_box()

    def bad_score(context):
        return np.ones((len(context.X), 3))

    with pytest.raises(ValueError, match="returned shape"):
        box.predict(
            score=bad_score,
            map_kwargs={"load_from_cache_file": False},
        )


def test_box_predict_single_member_rejected():
    box, _ = make_prediction_box(
        values=((1.0,),),
        weights=(1.0,),
    )

    with pytest.raises(ValueError, match="at least two"):
        box.predict(score="ensemble_sd")


def test_box_predict_unscored_unchanged():
    box, members = make_prediction_box()

    result = box.predict(
        batch_size=2,
        map_kwargs={"load_from_cache_file": False},
    )

    assert result.column_names == ["x", "prediction:y"]

    np.testing.assert_allclose(
        result["prediction:y"],
        np.full(5, 2.5),
    )

    assert all(m.calls == 3 for m in members.values())


def test_box_predict_batch_size_independence():
    box, _ = make_prediction_box()

    small = box.predict(
        score="ensemble_sd",
        batch_size=1,
        map_kwargs={"load_from_cache_file": False},
    )

    large = box.predict(
        score="ensemble_sd",
        batch_size=5,
        map_kwargs={"load_from_cache_file": False},
    )

    np.testing.assert_allclose(
        small["prediction:y"],
        large["prediction:y"],
    )

    np.testing.assert_allclose(
        small["score:ensemble_sd:y"],
        large["score:ensemble_sd:y"],
    )
