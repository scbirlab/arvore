"""Checkpoint construction-config round-trips and retraining."""

import json

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor

from treeton import Box
from treeton.config.instantiate import (
    load_training_config,
    resolve_experiment_config,
)


MODEL_SPEC = {
    "kind": "ensemble",
    "base": {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {
            "n_estimators": 8,
            "max_depth": 3,
            "random_state": 7,
        },
    },
    "n_members": 2,
    "seed": 42,
}

BOX_SPEC = {
    "model": MODEL_SPEC,
    "pipeline": {},
    "features": ["x"],
    "target": ["y"],
}


@pytest.fixture
def training_data():
    x = np.linspace(-2, 2, 40)
    return Dataset.from_dict({
        "x": x.tolist(),
        "y": (x ** 2 + 0.2 * x).tolist(),
    })


def test_box_construction_config_roundtrip():
    box = Box.from_config(BOX_SPEC)
    config = box.to_config()

    assert set(config) == {
        "model", "pipeline", "features", "target"
    }
    assert config["model"] == box.model.to_config()

    restored = Box.from_config(config)
    assert restored.to_config() == config
    assert len(restored.model.members) == 2


def test_box_to_config_is_unfitted_recipe(training_data):
    box = Box.from_config(BOX_SPEC)
    box.fit(training_data, skip_eval=True)

    fresh = Box.from_config(box.to_config())

    assert fresh is not box
    assert fresh.model is not box.model
    assert fresh.to_config() == box.to_config()

    for fitted, new in zip(
        box.model.members.values(),
        fresh.model.members.values(),
    ):
        assert hasattr(fitted, "estimators_")
        assert not hasattr(new, "estimators_")


def test_checkpoint_training_config_roundtrip(
    training_data, tmp_path
):
    box = Box.from_config(BOX_SPEC)
    box.fit(training_data, skip_eval=True)

    checkpoint = tmp_path / "original.tt"
    box.save(checkpoint)

    manifest = json.loads(
        (checkpoint / "config.json").read_text()
    )

    assert manifest["config"] == box.to_config()
    assert "model" in manifest["config"]
    assert "model_config" not in manifest["config"]

    experiment = load_training_config(checkpoint)

    assert set(experiment) == {"box"}
    assert experiment["box"] == box.to_config()

    fresh = Box.from_config(experiment["box"])
    assert fresh.to_config() == box.to_config()

    fresh.fit(training_data, skip_eval=True)
    next_checkpoint = tmp_path / "retrained.tt"
    fresh.save(next_checkpoint)

    assert load_training_config(next_checkpoint) == experiment


def test_checkpoint_manifest_and_standalone_config(
    training_data, tmp_path
):
    box = Box.from_config(BOX_SPEC)
    box.fit(training_data, skip_eval=True)

    checkpoint = tmp_path / "model.tt"
    box.save(checkpoint)

    assert load_training_config(
        checkpoint / "config.json"
    ) == {"box": box.to_config()}

    standalone = tmp_path / "box.json"
    box.to_config(standalone)

    assert load_training_config(standalone) == {
        "box": box.to_config()
    }


def test_existing_experiment_config_preserved():
    experiment = {
        "box": BOX_SPEC,
        "fit": {"skip_eval": True},
        "save": {"score": ["feature_distance"]},
    }

    assert load_training_config(experiment) == experiment


def test_checkpoint_model_override(training_data, tmp_path):
    box = Box.from_config(BOX_SPEC)
    box.fit(training_data, skip_eval=True)

    checkpoint = tmp_path / "model.tt"
    box.save(checkpoint)

    experiment = resolve_experiment_config(
        load_training_config(checkpoint),
        overrides=[
            "box.model.base.init_args.n_estimators=17",
        ],
    )

    fresh = Box.from_config(experiment["box"])

    assert all(
        member.n_estimators == 17
        for member in fresh.model.members.values()
    )


def test_opaque_checkpoint_requires_model_spec(
    training_data, tmp_path
):
    box = Box(
        model=RandomForestRegressor(
            n_estimators=8, random_state=42
        ),
        features=("x",),
        target=("y",),
        pipeline={},
    )
    box.fit(training_data, skip_eval=True)

    checkpoint = tmp_path / "opaque.tt"
    box.save(checkpoint)

    experiment = load_training_config(checkpoint)

    assert experiment["box"]["model"] is None

    with pytest.raises(ValueError, match="model specification"):
        Box.from_config(experiment["box"])


def test_checkpoint_loader_rejects_wrong_schema():
    with pytest.raises(ValueError, match="Expected an experiment"):
        load_training_config({"unexpected": 1})
