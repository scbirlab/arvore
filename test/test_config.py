"""Tests for declarative estimator configuration."""

from copy import deepcopy

import pytest
from sklearn.ensemble import RandomForestRegressor

from arvore.config.instantiate import (
    apply_overrides,
    instantiate_object,
    resolve_experiment_config,
)


def test_apply_overrides():
    """Dotted paths support JSON values and new configuration keys."""
    config = {
        "box": {
            "model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor",
                "init_args": {
                    "n_estimators": 100,
                    "random_state": 42,
                },
            },
            "features": ["fingerprint"],
            "target": "MIC",
        },
        "fit": {
            "verbose": False,
        },
    }

    original = deepcopy(config)

    observed = apply_overrides(
        config,
        [
            "box.model.init_args.n_estimators=500",
            "box.model.init_args.max_depth=8",
            "box.features=[\"fingerprint\",\"physchem\"]",
            "fit.verbose=true",
            "new.threshold=0.25",
            "new.enabled=false",
            "new.optional=null",
            "new.name=experiment",
        ],
    )

    assert observed["box"]["model"]["init_args"] == {
        "n_estimators": 500,
        "random_state": 42,
        "max_depth": 8,
    }
    assert observed["box"]["features"] == [
        "fingerprint", "physchem"
    ]
    assert observed["box"]["target"] == "MIC"

    assert observed["fit"]["verbose"] is True
    assert observed["new"]["threshold"] == pytest.approx(0.25)
    assert observed["new"]["enabled"] is False
    assert observed["new"]["optional"] is None
    assert observed["new"]["name"] == "experiment"

    # Configuration overrides must not mutate the caller's object.
    assert config == original


def test_apply_overrides_none():
    """No overrides preserve the configuration."""
    config = {"box": {"features": ["x"]}}

    observed = apply_overrides(config, None)

    assert observed == config
    assert observed is not config
    assert observed["box"] is not config["box"]


@pytest.mark.parametrize(
    "override",
    [
        "invalid",
        "=100",
        ".box=100",
        "box..features=100",
        "box.features.=100",
    ],
)
def test_apply_overrides_rejects_invalid_paths(override):
    with pytest.raises(ValueError):
        apply_overrides({}, [override])


def test_apply_overrides_rejects_scalar_intermediate():
    config = {
        "box": {
            "model": "random_forest",
        },
    }

    with pytest.raises(ValueError, match="not a mapping"):
        apply_overrides(
            config,
            ["box.model.init_args.n_estimators=100"],
        )


def test_apply_overrides_last_value_wins():
    observed = apply_overrides(
        {},
        [
            "box.model.init_args.n_estimators=100",
            "box.model.init_args.n_estimators=250",
        ],
    )

    assert (
        observed["box"]["model"]["init_args"]["n_estimators"]
        == 250
    )


def test_resolve_experiment_config():
    """Experiment config resolves dotted overrides before training."""
    config = {
        "box": {
            "model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor",
                "init_args": {
                    "n_estimators": 100,
                    "random_state": 42,
                },
            },
            "features": ["fingerprint"],
            "target": "MIC",
        },
    }

    observed = resolve_experiment_config(
        config,
        overrides=[
            "box.model.init_args.n_estimators=500",
            "box.features=[\"fingerprint\",\"physchem\"]",
        ],
    )

    assert (
        observed["box"]["model"]["init_args"]["n_estimators"]
        == 500
    )
    assert observed["box"]["features"] == [
        "fingerprint", "physchem"
    ]

    # The original configuration remains unchanged.
    assert config["box"]["model"]["init_args"]["n_estimators"] == 100


def test_instantiate_estimator():
    """Declarative model specs instantiate sklearn-compatible objects."""
    config = {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {
            "n_estimators": 50,
            "max_depth": 8,
            "random_state": 42,
        },
    }

    model = instantiate_object(config)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 50
    assert model.max_depth == 8
    assert model.random_state == 42
