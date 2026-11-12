
"""Tests for immutable model configuration and manifests."""

from dataclasses import FrozenInstanceError

import pytest

from treeton.config import (
    EstimatorConfig,
    EnsembleConfig,
    ModelArtifact,
    EnsembleManifest,
    parse_model_config,
)


BASE = {
    "class_path": "sklearn.ensemble.RandomForestRegressor",
    "init_args": {"n_estimators": 10},
}


def test_estimator_preserves_specification():
    config = EstimatorConfig.from_dict(BASE)

    assert config.to_dict(raw=True) == BASE
    assert config.class_path == BASE["class_path"]

    with pytest.raises(TypeError):
        config.init_args["n_estimators"] = 100

    with pytest.raises(FrozenInstanceError):
        config.class_path = "other"

    result = config.to_dict()
    result["init_args"]["n_estimators"] = 999
    assert config.to_dict(raw=True) == BASE


def test_omitted_defaults_remain_omitted():
    raw = {"kind": "ensemble", "base": BASE}
    config = EnsembleConfig.from_dict(raw)

    assert config.n_members == 1
    assert config.seed == 42
    assert config.to_dict(raw=True) == raw


def test_homogeneous_configuration():
    raw = {
        "kind": "ensemble",
        "schema_version": 1,
        "base": BASE,
        "n_members": 3,
        "seed": 12,
        "weights": [1, 2, 3],
    }

    config = EnsembleConfig.from_dict(raw)

    assert config.n_members == 3
    assert config.weights == (1, 2, 3)
    assert config.to_dict() == raw


def test_heterogeneous_configuration():
    raw = {
        "kind": "ensemble",
        "members": {
            "rf": BASE,
            "extra": {
                "class_path":
                    "sklearn.ensemble.ExtraTreesRegressor"
            },
        },
    }

    config = EnsembleConfig.from_dict(raw)

    assert len(config.members) == 2
    assert config.to_dict(raw=True) == raw

    with pytest.raises(TypeError):
        config.members["new"] = config.members["rf"]


@pytest.mark.parametrize("invalid", [
    {"kind": "ensemble"},
    {"kind": "ensemble", "base": BASE, "members": {}},
    {"kind": "ensemble", "base": BASE, "n_members": 0},
    {"kind": "ensemble", "base": BASE, "seed": True},
    {"kind": "ensemble", "base": BASE, "weights": [0]},
    {"kind": "ensemble", "base": BASE, "weights": [1, 2]},
    {"kind": "ensemble", "members": {"": BASE}},
    {"kind": "ensemble", "members": {}, "seed": 5},
])
def test_invalid_ensembles(invalid):
    with pytest.raises((ValueError, TypeError, KeyError)):
        EnsembleConfig.from_dict(invalid)


def test_single_estimator_compatibility():
    parsed = parse_model_config(BASE)

    assert isinstance(parsed, EstimatorConfig)
    assert parsed.to_dict(raw=True) == BASE


def test_artifact_validation():
    raw = {
        "adapter": "xgboost",
        "filename": "model.ubj",
        "class_path": "xgboost.sklearn.XGBRegressor",
        "version": "3.0",
        "sha256": "a" * 64,
        "trainable": True,
    }

    artifact = ModelArtifact.from_dict(raw)
    assert artifact.to_dict() == raw

    with pytest.raises(ValueError):
        ModelArtifact.from_dict({
            **raw, "filename": "../model.ubj"
        })

    with pytest.raises(ValueError):
        ModelArtifact.from_dict({
            **raw, "sha256": "invalid"
        })


def test_manifest_roundtrip():
    artifact = ModelArtifact(
        adapter="skops",
        filename="model.skops",
        class_path=BASE["class_path"],
        version="0.13",
        sha256="a" * 64,
        trainable=True,
    )

    manifest = EnsembleManifest(
        schema_version=1,
        config=EnsembleConfig.from_dict({
            "kind": "ensemble",
            "base": BASE,
            "n_members": 2,
        }),
        weights=(1.0, 2.0),
        members=(
            ("first", artifact),
            ("second", artifact),
        ),
    )

    restored = EnsembleManifest.from_dict(
        manifest.to_dict()
    )

    assert restored == manifest
    assert restored.config.to_dict()["n_members"] == 2
