import json
import subprocess

import numpy as np
from datasets import Dataset

from arvore import Box


def make_data(tmp_path):
    training = tmp_path / "train.parquet"
    testing = tmp_path / "test.parquet"

    Dataset.from_dict({
        "x": [float(i) for i in range(1, 31)],
        "y": [float(i * 2 + 1) for i in range(1, 31)],
    }).to_parquet(str(training))

    Dataset.from_dict({
        "x": [1.5, 6.5, 15.5],
    }).to_parquet(str(testing))

    return training, testing


def make_config():
    return {
        "model": {
            "class_path": "sklearn.ensemble.RandomForestRegressor",
            "init_args": {
                "n_estimators": 20,
                "random_state": 42,
            },
        },
        "pipeline": {
            "column_transforms": {
                "log_x": ["x", "log"],
            },
        },
        "features": ["log_x"],
        "target": "y",
    }


def test_box_checkpoint_roundtrip(tmp_path):
    from providence.io import load_json
    training, testing = make_data(tmp_path)

    box = Box.from_config(make_config())
    box.fit(str(training))

    before = np.asarray(
        box.predict(str(testing))["prediction:y"]
    )

    checkpoint = tmp_path / "model.ao"
    box.save(checkpoint)

    assert (checkpoint / "config.json").is_file()

    manifest = load_json(checkpoint / "config.json")
    assert (checkpoint / manifest["model"]["filename"]).is_file()
    assert (checkpoint / "training-data.as").is_dir()

    restored = Box.load(
        checkpoint,
        trusted_types=("sklearn.tree._tree.Tree",),
    )

    after = np.asarray(
        restored.predict(str(testing))["prediction:y"]
    )

    np.testing.assert_allclose(before, after)


def test_cli_train_predict(tmp_path):
    training, testing = make_data(tmp_path)

    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps({
        "box": make_config(),
    }))

    checkpoint = tmp_path / "model.ao"
    output = tmp_path / "predictions.parquet"

    subprocess.run(
        [
            "arvore", "train", str(training),
            "--config", str(config_path),
            "--checkpoint", str(checkpoint),
        ],
        check=True,
    )

    subprocess.run(
        [
            "arvore", "predict", str(checkpoint),
            "--trust-sklearn-trees",
            "--data", str(testing),
            "--output", str(output),
        ],
        check=True,
    )

    predictions = Dataset.from_parquet(str(output))

    assert "prediction:y" in predictions.column_names
    assert len(predictions) == 3
    assert np.isfinite(
        predictions["prediction:y"]
    ).all()



import pytest


@pytest.mark.parametrize(
    "class_path,init_args,dependency",
    [
        (
            "sklearn.ensemble.RandomForestRegressor",
            {"n_estimators": 10, "random_state": 42},
            "sklearn",
        ),
        (
            "xgboost.XGBRegressor",
            {
                "n_estimators": 30,
                "n_jobs": 1,
                "random_state": 42,
            },
            "xgboost",
        ),
        (
            "lightgbm.LGBMRegressor",
            {
                "n_estimators": 30,
                "min_child_samples": 2,
                "verbosity": -1,
                "n_jobs": 1,
            },
            "lightgbm",
        ),
        # (
        #     "catboost.CatBoostRegressor",
        #     {
        #         "iterations": 30,
        #         "verbose": False,
        #         "thread_count": 1,
        #     },
        #     "catboost",
        # ),
    ],
)
def test_backend_box_fit(
    tmp_path, class_path, init_args, dependency
):
    pytest.importorskip(dependency)

    training, testing = make_data(tmp_path)

    config = make_config()
    config["model"] = {
        "class_path": class_path,
        "init_args": init_args,
    }

    box = Box.from_config(config)
    box.fit(str(training))

    predictions = np.asarray(
        box.predict(str(testing))["prediction:y"]
    )

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_box_model_config_delegation():
    from sklearn.ensemble import RandomForestRegressor
    from arvore import Ensemble

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

    model = Ensemble.from_config(config)

    box = Box(
        model=model,
        features=["x"],
        target="y",
    )

    assert box.model_config == config
    assert box.to_config()["model"] == config

    # No duplicated configuration in Box.

    # Defensive copying.
    returned = box.model_config
    returned["base"]["init_args"]["n_estimators"] = 999

    assert model.to_config() == config
    assert box.model_config == config


def test_restored_model_config(tmp_path):
    from datasets import Dataset

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

    data = Dataset.from_dict({
        "x": [0., 1., 2., 3., 4.],
        "y": [0., 1., 4., 9., 16.],
    })

    box = Box.from_config(config)
    box.fit(data, skip_eval=True)
    box.save(tmp_path / "model")

    restored = Box.load(
        tmp_path / "model",
        trusted_types=("sklearn.tree._tree.Tree",),
    )

    assert restored.model.to_config() == config["model"]
    assert restored.model_config == config["model"]


def test_single_estimator_is_one_member_ensemble(tmp_path):
    from sklearn.ensemble import RandomForestRegressor
    from arvore import Ensemble

    estimator = RandomForestRegressor(n_estimators=5, random_state=0)
    box = Box(model=estimator, features="x", target="y")

    assert isinstance(box.model, Ensemble)
    assert box.model.members == {"model": estimator}
    assert box.model_config is None


def test_legacy_single_estimator_checkpoint(tmp_path):
    from providence.io import load_json, save_json
    from arvore import Ensemble
    from arvore.models.adapters import save_model

    training, testing = make_data(tmp_path)
    box = Box.from_config(make_config())
    box.fit(str(training), skip_eval=True)
    checkpoint = tmp_path / "legacy.ao"
    box.save(checkpoint)

    # Checkpoints from before Box always held an Ensemble saved the bare estimator.
    manifest = load_json(checkpoint / "config.json")
    manifest["model"] = save_model(box.model.members["model"], checkpoint)
    save_json(manifest, checkpoint / "config.json")

    restored = Box.load(checkpoint, trusted_types=("sklearn.tree._tree.Tree",))

    assert isinstance(restored.model, Ensemble)
    np.testing.assert_allclose(
        restored.predict(str(testing))["prediction:y"],
        box.predict(str(testing))["prediction:y"],
    )


def test_schema_1_checkpoint_deprecated(tmp_path):
    import pytest
    from providence.io import save_json

    checkpoint = tmp_path / "schema-1.ao"
    checkpoint.mkdir()
    save_json(
        {"schema_version": 1, "artefacts": {"model_binary": "model.joblib"}},
        checkpoint / "config.json",
    )

    with pytest.warns(DeprecationWarning, match="Schema-1"):
        with pytest.raises(ValueError, match="joblib"):
            Box.load(checkpoint)


def test_bare_estimator_lifecycle(tmp_path):
    from sklearn.ensemble import RandomForestRegressor
    from arvore import Ensemble

    training, testing = make_data(tmp_path)
    estimator = RandomForestRegressor(n_estimators=10, random_state=0)
    box = Box(model=estimator, features="x", target="y", pipeline={})

    box.fit(str(training), skip_eval=True)
    predictions = np.asarray(box.predict(str(testing))["prediction:y"])

    assert hasattr(estimator, "estimators_")
    np.testing.assert_allclose(
        predictions,
        estimator.predict(np.array([[1.5], [6.5], [15.5]])),
    )
    assert box.to_config()["model"] is None

    box.save(tmp_path / "bare.ao")
    restored = Box.load(tmp_path / "bare.ao", trusted_types=("sklearn.tree._tree.Tree",))

    assert isinstance(restored.model, Ensemble)
    assert isinstance(restored.model.members["model"], RandomForestRegressor)
    np.testing.assert_allclose(
        np.asarray(restored.predict(str(testing))["prediction:y"]),
        predictions,
    )
    restored.fit(str(training), skip_eval=True)


def test_saved_references_restore_in_fresh_process(tmp_path):
    import sys

    training, testing = make_data(tmp_path)
    box = Box.from_config(make_config())
    box.fit(str(training), skip_eval=True)
    checkpoint = tmp_path / "scored.ao"
    box.save(checkpoint, score=["feature_distance", "inverse_leaf_occupancy"])

    # A fresh interpreter has imported nothing that registers reference types.
    script = (
        "from arvore import Box\n"
        f"box = Box.load({str(checkpoint)!r}, trusted_types=('sklearn.tree._tree.Tree',))\n"
        "assert len(box.references.snapshot()) == 2, box.references.snapshot()\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)

