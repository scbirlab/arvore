import json

import numpy as np
import pytest

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from arvore import Box, Ensemble
from arvore.models.adapters import save_model, load_model
from arvore.models.ensemble import Ensemble

TRUSTED_SKLEARN_TYPES = ("sklearn.tree._tree.Tree",)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] ** 2 + X[:, 1] - X[:, 2] * 0.5
    return X, y


def test_repeated_ensemble(regression_data):
    X, y = regression_data

    model = Ensemble.repeat(
        RandomForestRegressor(n_estimators=12, n_jobs=1),
        n=3,
        seed=10,
    )
    model.fit(X, y)

    members = model.predict(X, disaggregate=True)

    assert members.shape == (80, 3, 1)
    assert len(model) == 3
    assert [
        m.random_state for m in model.members.values()
    ] == [10, 11, 12]

    np.testing.assert_allclose(
        model.predict(X)[...,None],
        members.mean(axis=1),
    )


def test_weighted_heterogeneous_ensemble(regression_data):
    X, y = regression_data

    model = Ensemble(
        {
            "rf": RandomForestRegressor(
                n_estimators=12, random_state=1
            ),
            "extra": ExtraTreesRegressor(
                n_estimators=12, random_state=2
            ),
        },
        weights=[3, 1],
    )
    model.fit(X, y)

    members = model.predict(X, disaggregate=True)

    np.testing.assert_allclose(
        model.predict(X)[...,None],
        np.average(members, axis=1, weights=[3, 1]),
    )


def test_ensemble_codec_roundtrip(tmp_path, regression_data):
    X, y = regression_data

    model = Ensemble.repeat(
        RandomForestRegressor(n_estimators=12),
        n=2,
        seed=20,
    )
    model.fit(X, y)

    metadata = save_model(model, tmp_path)

    assert metadata["adapter"] == "ensemble"
    assert (tmp_path / "model.json").exists()
    assert (
        tmp_path / "members" / "0000" / "model.skops"
    ).exists()

    restored = load_model(
        tmp_path, 
        metadata,
        trusted_types=TRUSTED_SKLEARN_TYPES,
    )

    assert list(restored.members) == list(model.members)

    np.testing.assert_allclose(
        model.predict(X),
        restored.predict(X),
    )

    # Corruption of an individual member must be detected.
    member_file = (
        tmp_path / "members" / "0000" / "model.skops"
    )
    member_file.write_bytes(member_file.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum"):
        load_model(
            tmp_path, 
            metadata, 
            trusted_types=TRUSTED_SKLEARN_TYPES,
        )


def test_declarative_ensemble(regression_data):
    X, y = regression_data

    config = {
        "schema_version": 1,
        "kind": "ensemble",
        "base": {
            "class_path": "sklearn.ensemble.RandomForestRegressor",
            "init_args": {"n_estimators": 10},
        },
        "n_members": 3,
        "seed": 42,
    }

    model = Ensemble.from_config(config)
    model.fit(X, y)

    assert len(model) == 3
    assert model.predict(X).shape == (80,)


def test_box_checkpoint(tmp_path):
    from datasets import Dataset

    x = np.linspace(-3, 3, 50)
    data = Dataset.from_dict({
        "x": x.tolist(),
        "y": (x ** 2).tolist(),
    })

    config = {
        "model": {
            "kind": "ensemble",
            "schema_version": 1,
            "base": {
                "class_path": "sklearn.ensemble.RandomForestRegressor",
                "init_args": {"n_estimators": 10},
            },
            "n_members": 2,
            "seed": 42,
        },
        "features": ["x"],
        "target": ["y"],
        "pipeline": {},
    }

    box = Box.from_config(config)
    box.fit(data, skip_eval=True)

    before = box.predict(data)
    path = tmp_path / "box"
    box.save(path)

    restored = Box.load(
        path,
        trusted_types=TRUSTED_SKLEARN_TYPES,
    )
    after = restored.predict(data)

    np.testing.assert_allclose(
        before["prediction:y"],
        after["prediction:y"],
    )


def test_mixed_backend_codec(tmp_path, regression_data):
    xgb = pytest.importorskip("xgboost")
    X, y = regression_data

    model = Ensemble({
        "rf": RandomForestRegressor(
            n_estimators=10, random_state=1
        ),
        "xgb": xgb.XGBRegressor(
            n_estimators=10, random_state=2, n_jobs=1
        ),
    })
    model.fit(X, y)

    metadata = save_model(model, tmp_path)
    manifest = json.loads(
        (tmp_path / "model.json").read_text()
    )

    assert [
        member["metadata"]["adapter"]
        for member in manifest["members"]
    ] == ["skops", "xgboost"]

    assert (
        tmp_path / "members" / "0001" / "model.ubj"
    ).exists()

    restored = load_model(
        tmp_path, 
        metadata, 
        trusted_types=TRUSTED_SKLEARN_TYPES,
    )

    np.testing.assert_allclose(
        model.predict(X),
        restored.predict(X),
    )
    

def test_ensemble_config_roundtrip():
    config = {
        "kind": "ensemble",
        "schema_version": 1,
        "base": {
            "class_path": "sklearn.ensemble.RandomForestRegressor",
            "init_args": {"n_estimators": 10},
        },
        "n_members": 3,
        "seed": 42,
    }

    ensemble = Ensemble.from_config(config)

    assert ensemble.to_config() == config

    # Returned configurations must not be live references.
    restored = ensemble.to_config()
    restored["base"]["init_args"]["n_estimators"] = 999

    assert ensemble.to_config() == config


def test_single_estimator_config_roundtrip():
    config = {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {"n_estimators": 10},
    }

    ensemble = Ensemble.from_config(config)

    assert len(ensemble) == 1
    assert ensemble.to_config() == config


def test_codec_rountrip(tmp_path, regression_data):
    config = {
        "kind": "ensemble",
        "schema_version": 1,
        "base": {
            "class_path": "sklearn.ensemble.RandomForestRegressor",
            "init_args": {"n_estimators": 10},
        },
        "n_members": 2,
        "seed": 42,
    }
    X, y = regression_data

    model = Ensemble.from_config(config)
    model.fit(X, y)

    metadata = save_model(model, tmp_path)
    restored = load_model(
        tmp_path,
        metadata,
        trusted_types=("sklearn.tree._tree.Tree",),
    )

    assert restored.to_config() == config
    np.testing.assert_allclose(model.predict(X), restored.predict(X))
