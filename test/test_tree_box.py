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
    training, testing = make_data(tmp_path)

    box = Box.from_config(make_config())
    box.fit(str(training))

    before = np.asarray(
        box.predict(str(testing))["prediction:y"]
    )

    checkpoint = tmp_path / "model.tt"
    box.save(checkpoint)

    assert (checkpoint / "config.json").is_file()
    assert (checkpoint / "model.joblib").is_file()
    assert (checkpoint / "training-data.as").is_dir()

    restored = Box.load(checkpoint)

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
