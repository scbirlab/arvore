"""Canonical scorer registry and naming tests."""

import pytest

from treeton.scorers import (
    EnsembleSD,
    TanimotoDistance,
    TreeSD,
    resolve_scorers,
)
from treeton.scorers.runner import ScoringRunner


def test_none_resolves_to_empty_mapping():
    assert resolve_scorers(None) == {}


def test_registered_names_resolve_to_instances():
    resolved = resolve_scorers([
        "ensemble_sd",
        "tree_sd",
    ])

    assert list(resolved) == [
        "ensemble_sd",
        "tree_sd",
    ]

    assert isinstance(
        resolved["ensemble_sd"],
        EnsembleSD,
    )
    assert isinstance(
        resolved["tree_sd"],
        TreeSD,
    )


def test_explicit_alias_preserves_identity():
    first = TanimotoDistance(column="fp1")
    second = TanimotoDistance(column="fp2")

    resolved = resolve_scorers({
        "morgan_1": first,
        "morgan_2": second,
    })

    assert resolved == {
        "morgan_1": first,
        "morgan_2": second,
    }

    assert resolved["morgan_1"] is first
    assert resolved["morgan_2"] is second

    # Explicit output names do not modify scorers.
    assert first.name == "tanimoto_distance"
    assert second.name == "tanimoto_distance"


def test_mapping_accepts_registered_names():
    resolved = resolve_scorers({
        "disagreement": "ensemble_sd",
        "within_forest": "tree_sd",
    })

    assert isinstance(
        resolved["disagreement"],
        EnsembleSD,
    )
    assert isinstance(
        resolved["within_forest"],
        TreeSD,
    )


def test_duplicate_implicit_names_rejected():
    with pytest.raises(
        ValueError,
        match="Duplicate scorer output name",
    ):
        resolve_scorers([
            TanimotoDistance(column="fp1"),
            TanimotoDistance(column="fp2"),
        ])


def test_invalid_alias_rejected():
    with pytest.raises(
        ValueError,
        match="nonempty strings",
    ):
        resolve_scorers({
            "": TreeSD(),
        })


def test_scorer_class_not_accepted():
    with pytest.raises(TypeError, match="instance"):
        resolve_scorers(TreeSD)


def test_preparation_state_keyed_by_alias():
    class DummyScorer:
        name = "dummy"

        def __init__(self, value):
            self.value = value

        def prepare(self, box, *, batch_size):
            return {
                "value": self.value,
                "batch_size": batch_size,
            }

        def __call__(self, context):
            return context.state["value"]

    first = DummyScorer(1)
    second = DummyScorer(2)

    runner = ScoringRunner(
        box=object(),
        scorers={
            "first": first,
            "second": second,
        },
    )

    runner.prepare(batch_size=16)

    assert runner.states == {
        "first": {
            "value": 1,
            "batch_size": 16,
        },
        "second": {
            "value": 2,
            "batch_size": 16,
        },
    }

    assert runner.scorers["first"] is first
    assert runner.scorers["second"] is second
