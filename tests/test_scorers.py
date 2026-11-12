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
