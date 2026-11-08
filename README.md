
# 🌳 Treeton

[GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/treeton/python-test.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/treeton)
![PyPI](https://img.shields.io/pypi/v/treeton)

**Treeton** is a lightweight framework for reproducible tree-based machine learning, combining declarative model configuration, data featurisation, provenance tracking, training, evaluation and checkpointing.

Treeton works with estimators implementing the familiar scikit-learn `fit(X, y)` and `predict(X)` interface. It delegates data loading and provenance to [Providence](https://github.com/scbirlab/providence) and feature transformations to [Aspect](https://github.com/scbirlab/aspect).

The goal is to make a machine-learning experiment easy to **specify, run, save and reproduce**, whether the input is a local table, molecular fingerprints or other structured scientific data.

## Features

- **Estimator-agnostic training** — instantiate compatible estimators from a JSON configuration, including scikit-learn tree models.
- **Composable feature transformations** — use Aspect to generate, concatenate and reuse features.
- **Dataset provenance** — resolve local and remote datasets using Providence, with source metadata carried into training checkpoints.
- **Checkpointing** — save fitted models together with pipeline configurations and training-data artefacts.
- **Batched prediction** — predict over Hugging Face Datasets, with optional CLI row selection.
- **Model evaluation** — calculate RMSE/RMSD, MAE, Pearson's *r* and Spearman's ρ.
- **CLI and Python API** — use the same `Box` abstraction interactively or in automated workflows.

Treeton is at an early stage of development. Random-forest regression is covered by end-to-end tests; compatibility with other estimators depends on their APIs.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line interface](#command-line-interface)
- [Configuration](#configuration)
- [Python API](#python-api)
- [Molecular machine learning](#molecular-machine-learning)
- [Checkpoints and reproducibility](#checkpoints-and-reproducibility)
- [Development](#development)
- [Related packages](#related-packages)

## Installation

Treeton requires Python 3.11 or later.

### From GitHub

```bash
pip install "git+https://github.com/scbirlab/treeton.git@dev"
```

For development:

```bash
git clone https://github.com/scbirlab/treeton.git
cd treeton
pip install -e ".[dev]"
```

### Optional dependencies

Additional packages are available for specialised workflows:

```bash
# Chemical featurisation through Schemist
pip install -e ".[chem]"

# Biological sequence and taxonomy features through Vectome
pip install -e ".[bio]"

# XGBoost
pip install -e ".[xgb]"

# LightGBM
pip install -e ".[lgbm]"
```

See `pyproject.toml` for the complete list of optional dependency groups, including CUDA-specific XGBoost options.

## Quick start

Train a random forest, save it and predict on new observations using a single JSON configuration.

### 1. Prepare some data

For a self-contained example, create a small CSV training dataset and a prediction dataset:

```bash
printf 'x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n9,19\n10,21\n' > train.csv

printf 'x\n2.5\n5.5\n8.5\n' > unseen.csv
```

### 2. Define an experiment

Save the following as `experiment.json`:

```json
{
  "box": {
    "model": {
      "class_path": "sklearn.ensemble.RandomForestRegressor",
      "init_args": {
        "n_estimators": 100,
        "random_state": 42
      }
    },
    "pipeline": {
      "column_transforms": {
        "log_x": ["x", "log"]
      }
    },
    "features": ["log_x"],
    "target": "y"
  }
}
```

This configuration specifies:

1. A `RandomForestRegressor` with reproducible initialisation.
2. An Aspect transform generating `log_x` from `x`.
3. The feature columns supplied to the estimator.
4. The prediction target.

Treeton converts the selected feature columns into a numeric matrix before calling the estimator.

### 3. Train

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint model.tt
```

The `model.tt` directory contains the fitted estimator and the information required to reconstruct its preprocessing pipeline.

### 4. Predict

```bash
treeton predict model.tt \
    --data unseen.csv \
    --output predictions.parquet
```

The output contains the input observations and a prediction column named `prediction:y`.

You can also save the predictions as a Providence dataset checkpoint:

```bash
treeton predict model.tt \
    --data unseen.csv \
    --checkpoint predictions.pv
```

Both output options can be used together.

## Command-line interface

Treeton provides two primary commands:

```bash
treeton train --help
treeton predict --help
```

### Training

```bash
treeton train DATA \
    --config experiment.json \
    --checkpoint model.tt
```

`DATA` can be a supported local or remote dataset specification understood by Providence.

Common training options:

| Option | Description |
| --- | --- |
| `--config` | JSON experiment configuration |
| `--checkpoint`, `-c` | Output model checkpoint directory |
| `--validation`, `-2` | Optional validation dataset |
| `--pipeline`, `-p` | Override the configured Aspect pipeline with a JSON file |
| `--set` | Override a configuration value |
| `--cache` | Directory for dataset and preprocessing caches |

#### Configuration overrides

Use `--set` to override nested configuration values without editing the JSON file:

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint model.tt \
    --set box.model.init_args.n_estimators=500 \
    --set box.model.init_args.max_depth=12
```

Overrides use `dotted.path=value` notation. Values are parsed as JSON when possible, so integers, floating-point numbers, booleans, `null` and arrays are supported.

For example:

```bash
--set box.features='["log_x","another_feature"]'
```

The corresponding feature must exist in the input data or be generated by the selected pipeline.

### Prediction

```bash
treeton predict MODEL_CHECKPOINT \
    --data DATA \
    --output predictions.parquet
```

Common prediction options:

| Option | Description |
| --- | --- |
| `--data`, `-d` | Dataset on which to predict |
| `--output`, `-o` | Output table, e.g. CSV or Parquet |
| `--checkpoint`, `-c` | Output Providence dataset checkpoint |
| `--start` | First row to predict, inclusive |
| `--stop` | Row at which to stop, exclusive |
| `--pipeline`, `-p` | Override preprocessing pipeline |
| `--config` | Optional JSON prediction settings |
| `--set` | Override prediction settings |
| `--cache` | Cache directory |

For example, predict a subset of a larger dataset:

```bash
treeton predict model.tt \
    --data dataset.parquet \
    --start 10000 \
    --stop 20000 \
    --output predictions-10000-20000.parquet
```

The selected interval follows Python slicing semantics: `[start:stop]`.

Prediction settings may also be supplied in JSON:

```json
{
  "batch_size": 2048
}
```

```bash
treeton predict model.tt \
    --data unseen.csv \
    --config predict.json \
    --output predictions.parquet
```

## Configuration

The experiment configuration has a small number of top-level sections:

```json
{
  "box": {
    "model": {
      "class_path": "sklearn.ensemble.RandomForestRegressor",
      "init_args": {
        "n_estimators": 100,
        "random_state": 42
      }
    },
    "pipeline": {},
    "features": ["feature_1", "feature_2"],
    "target": "response"
  },
  "fit": {},
  "save": {}
}
```

### `box.model`

The model is specified using a fully qualified Python class path and constructor arguments.

```json
{
  "class_path": "sklearn.ensemble.ExtraTreesRegressor",
  "init_args": {
    "n_estimators": 200,
    "random_state": 42
  }
}
```

This mechanism can instantiate other compatible estimator classes available in the environment. Backend-specific behaviour may require additional integration and testing.

### `box.pipeline`

The pipeline is an [Aspect](https://github.com/scbirlab/aspect) configuration describing transformations from source columns to model features.

For example:

```json
{
  "column_transforms": {
    "log_x": ["x", "log"]
  }
}
```

The `features` list refers to the logical feature names, including transformed output names.

A model can use both existing and transformed columns:

```json
{
  "features": ["numeric_feature", "log_x"]
}
```

### `box.features` and `box.target`

Features are supplied as an ordered list of column names. Each selected column can contain a scalar or a fixed-length numeric vector; Treeton concatenates them into a numeric feature matrix.

The target may be a single column or a list of columns:

```json
{
  "features": ["fingerprint", "descriptors"],
  "target": ["activity_a", "activity_b"]
}
```

Multi-target training requires an estimator that supports multiple outputs.

### `fit` and `save`

The `fit` section passes keyword arguments to `Box.fit()`. The `save` section passes options to the underlying Aspect pipeline checkpoint.

For example:

```json
{
  "save": {
    "save_source_data": true,
    "save_transformed": false
  }
}
```

Whether the source data are embedded or referenced affects checkpoint size and portability.

## Python API

The core abstraction is `treeton.Box`, which combines an estimator, its feature definitions and an Aspect pipeline.

### Train and predict

```python
import json

from treeton import Box

with open("experiment.json") as f:
    experiment = json.load(f)

box = Box.from_config(experiment["box"])

box.fit("train.csv")

predictions = box.predict("unseen.csv")
print(predictions["prediction:y"])
```

`predict()` returns a Hugging Face Dataset containing prediction columns alongside the prepared input data.

### Save and restore

```python
box.save("model.tt")

restored = Box.load("model.tt")

predictions = restored.predict("unseen.csv")
```

The model and preprocessing pipeline are restored together.

### Use an existing estimator

You can also construct a `Box` directly:

```python
from sklearn.ensemble import RandomForestRegressor
from treeton import Box

box = Box(
    model=RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    ),
    features=["x"],
    target="y",
)

box.fit("train.csv")
box.save("model.tt")
```

A model passed directly need only implement compatible `fit()` and `predict()` methods.

### Evaluate a model

```python
predicted, scores = box.evaluate(
    "train.csv",
    metrics=["rmse", "mae", "pearson", "spearman"],
)

print(scores)
```

`evaluate()` returns a tuple containing the predicted dataset and a dictionary of metric values.

Available registered regression metrics include:

| Metric | Description |
| --- | --- |
| `rmsd`, `rmse` | Root mean square deviation/error |
| `mae` | Mean absolute error |
| `pearson` | Pearson correlation coefficient |
| `spearman` | Spearman rank correlation coefficient |

Evaluation requires the target columns to be present. For meaningful model assessment, use held-out data rather than the training set.

### Verify the training data source

When the original data source is available through Providence, its recorded provenance can be accessed and verified:

```python
source = box.training_data_source

if source is not None:
    print(box.verify_training_data_source())
```

A returned `None` indicates that no verifiable source is available. Source verification does not independently guarantee reproducibility of the entire software environment.

## Molecular machine learning

Treeton is intended to work with scientific feature representations, including molecular fingerprints and descriptors.

For chemical featurisation, install the `chem` optional dependencies and specify Aspect transforms:

```json
{
  "box": {
    "model": {
      "class_path": "sklearn.ensemble.RandomForestRegressor",
      "init_args": {
        "n_estimators": 200,
        "random_state": 42
      }
    },
    "pipeline": {
      "column_transforms": {
        "fp": ["smiles", "morgan-fingerprint"],
        "descriptors": ["smiles", "descriptors-2d"]
      }
    },
    "features": ["fp", "descriptors"],
    "target": "activity"
  }
}
```

This example assumes a dataset with `smiles` and `activity` columns. The transforms create feature vectors that Treeton concatenates before fitting the model.

Dataset specifications supported by Providence can also refer to remote resources, including Hugging Face datasets:

```bash
treeton train \
    'hf://scbirlab/fang-2023-biogen-adme~scaffold-split:train' \
    --config experiment.json \
    --checkpoint model.tt
```

Adjust the feature and target definitions to match the selected dataset.

For dataset splitting, see [Eluent](https://github.com/scbirlab/eluent), which handles split assignments separately from model training.

## Checkpoints and reproducibility

A Treeton model checkpoint is a directory containing the saved estimator, a manifest and its Aspect pipeline checkpoint.

For example:

```text
model.tt/
├── config.json
├── model.joblib
└── training-data.as/
    └── ...
```

`config.json` records the checkpoint schema, experiment configuration and artefact locations.

`model.joblib` stores the fitted Python estimator. Models exposing a native `save()` method may also generate a native-format artefact, although the current loader restores the joblib representation.

`training-data.as/` contains the Aspect pipeline checkpoint, including its configuration and the training-data source information managed by Providence.

The pipeline can preserve references to immutable remote datasets or package local source data, according to its checkpoint options.

### Reproducibility considerations

A checkpoint makes model restoration convenient, but exact reproducibility also depends on:

- The original data and its version or checksum.
- The preprocessing implementation and dependencies.
- The fitted model's software versions.
- Random seeds and numerical execution environment.

For long-term archival, retain the checkpoint alongside a record of the package environment, such as a lock file or container image.

**Security:** joblib uses pickle-based deserialization. Only load model checkpoints from trusted sources.

## Development

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the unit and integration tests:

```bash
pytest -q test
```

Run module doctests:

```bash
pytest -q treeton --doctest-modules
```

The GitHub Actions workflows also exercise the command-line interface, including training and prediction on a remote scientific dataset.

To report a bug or propose a feature, open an [issue](https://github.com/scbirlab/treeton/issues).

## Related packages

Treeton is designed to work as part of a modular machine-learning workflow:

| Package | Role |
| --- | --- |
| [Providence](https://github.com/scbirlab/providence) | Dataset resolution, provenance and checkpointing |
| [Eluent](https://github.com/scbirlab/eluent) | Dataset splitting and split provenance |
| [Aspect](https://github.com/scbirlab/aspect) | Feature transformation and representation |
| [Treeton](https://github.com/scbirlab/treeton) | Tree-based estimator training and evaluation |
| [Duvidnn](https://github.com/scbirlab/duvidnn) | Neural-network modelling and uncertainty methods |

Each package has a separate responsibility, making it possible to reuse the same data sources, splits and feature representations across different model families.

## License

Treeton is distributed under the [MIT License](LICENSE).
