import json
import subprocess

import numpy as np
from datasets import Dataset

from treeton import Box


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

    checkpoint = tmp_path / "model.tt"
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

    checkpoint = tmp_path / "model.tt"
    output = tmp_path / "predictions.parquet"

    subprocess.run(
        [
            "treeton", "train", str(training),
            "--config", str(config_path),
            "--checkpoint", str(checkpoint),
        ],
        check=True,
    )

    subprocess.run(
        [
            "treeton", "predict", str(checkpoint),
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
    assert box.to_config()["model_config"] == config

    # No duplicated configuration in Box.
    assert box._model_config_fallback is None

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
    assert restored._model_config_fallback is None
