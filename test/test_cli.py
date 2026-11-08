import json
from argparse import Namespace

import pandas as pd

from treeton.cli.predict import _predict
from treeton.cli.train import _train
from treeton.config import (
    apply_overrides,
    resolve_experiment_config,
)


def test_apply_overrides():

    config = {
        "trainer": {
            "max_epochs": 10,
            "optimizer_kwargs": {
                "lr": 0.01,
            },
        },
        "fit": {
            "batch_size": 32,
        },
    }

    observed = apply_overrides(
        config,
        [
            "trainer.max_epochs=100",
            "trainer.optimizer_kwargs.lr=0.001",
            "fit.batch_size=64",
            "new.flag=true",
            "new.name=test",
        ],
    )

    assert observed["trainer"]["max_epochs"] == 100
    assert observed["trainer"]["optimizer_kwargs"]["lr"] == 0.001
    assert observed["fit"]["batch_size"] == 64
    assert observed["new"]["flag"] is True
    assert observed["new"]["name"] == "test"


def test_model_alias():

    config = {
        "box": {
            "model": {
                "class_path": "torch.nn.Identity"
            },
            "pipeline": {},
            "input_map": {},
        }
    }

    observed = resolve_experiment_config(
        config,
        model="mlp",
        overrides=[
            "box.model.init_args.output_dim=2",
        ],
    )

    assert (
        observed["box"]["model"]["class_path"]
        == "duvidnn.models.MLP"
    )

    assert (
        observed["box"]["model"]["init_args"]["output_dim"]
        == 2
    )


def test_config_cli_train_predict(tmp_path):
    training = tmp_path / "training.csv"
    prediction = tmp_path / "prediction.csv"
    config_file = tmp_path / "config.json"
    checkpoint = tmp_path / "model"
    output = tmp_path / "predictions.parquet"

    pd.DataFrame({
        "x": [
            [0.],
            [1.],
            [2.],
            [3.],
        ],
        "y": [
            [0.],
            [2.],
            [4.],
            [6.],
        ],
    }).to_parquet(
        training.with_suffix(".parquet"),
        index=False,
    )

    training = training.with_suffix(".parquet")

    pd.DataFrame({
        "x": [
            [4.],
            [5.],
        ],
    }).to_parquet(
        prediction.with_suffix(".parquet"),
        index=False,
    )

    prediction = prediction.with_suffix(".parquet")

    config = {
        "box": {
            "model": {
                "class_path": "torch.nn.Linear",
                "init_args": {
                    "in_features": 1,
                    "out_features": 1,
                },
            },
            "pipeline": {},
            "input_map": {
                "inputs": {
                    "input": "x",
                },
                "target": "y",
            },
        },
        "trainer": {
            "max_epochs": 2,
            "loss": {
                "class_path": "torch.nn.MSELoss",
            },
            "optimizer": "torch.optim.Adam",
            "optimizer_kwargs": {"lr": .01},
            "logger": False,
            "enable_checkpointing": False,
            "enable_model_summary": False,
        },
        "fit": {"batch_size": 2},
    }

    config_file.write_text(json.dumps(config))

    _train(
        Namespace(
            config=str(config_file),
            training=str(training),
            validation=None,
            output=str(checkpoint),
            model=None,
            set=[
                "trainer.max_epochs=1",
                "fit.batch_size=2",
            ],
            cache=str(tmp_path / "cache"),
        )
    )

    assert (checkpoint / "config.json").exists()

    _predict(
        Namespace(
            checkpoint=str(checkpoint),
            data=str(prediction),
            config=None,
            output=str(output),
            set=None,
            cache=str(tmp_path / "cache"),
        )
    )

    assert output.exists()
