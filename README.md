# 🌳 Treeton

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/treeton/python-test.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/treeton)
![PyPI](https://img.shields.io/pypi/v/treeton)

**Treeton** is a lightweight framework for reproducible tree-based machine learning, combining declarative model configuration, composable ensembles, data featurisation, provenance tracking, training, evaluation and checkpointing.

Treeton supports estimators implementing the familiar scikit-learn `fit(X, y)` and `predict(X)` interface. Models can be trained individually or combined into homogeneous or heterogeneous ensembles, with fitted members saved in their native formats where supported.

Treeton delegates data loading and provenance to [Providence](https://github.com/scbirlab/providence) and feature transformations to [Aspect](https://github.com/scbirlab/aspect).

The goal is to make a machine-learning experiment easy to **specify, run, save, share and reproduce**, whether the input is a local table, molecular fingerprints or other structured scientific data.

## Features

- **Estimator-agnostic training** — use sklearn-compatible estimators, including random forests, XGBoost, LightGBM and CatBoost.
- **Composable ensembles** — train repeated instances of one estimator or combine heterogeneous estimators under a common interface.
- **Member-level predictions** — obtain weighted aggregate predictions or individual member predictions for uncertainty and disagreement analysis.
- **Declarative configuration** — instantiate estimators and ensembles from JSON, including reproducible seeds, constructor arguments and ensemble weights.
- **Composable feature transformations** — use Aspect to generate, concatenate and reuse features.
- **Dataset provenance** — resolve local and remote datasets using Providence, with source metadata carried into training checkpoints.
- **Native model checkpointing** — serialise supported tree estimators through backend-specific codecs, preserving model state and integrity checks.
- **Batched prediction** — predict over Hugging Face Datasets, with optional CLI row selection.
- **Model evaluation** — calculate RMSE/RMSD, MAE, Pearson's *r* and Spearman's ρ.
- **CLI and Python API** — use the same `Box` abstraction interactively or in automated workflows.

Treeton is in early development. Its current focus is numeric regression. Individual estimator compatibility depends on the estimator's API, and not every compatible estimator has a supported safe serialization format.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line interface](#command-line-interface)
- [Configuration](#configuration)
- [Python API](#python-api)
- [Ensembles](#ensembles)
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

# CatBoost
pip install -e ".[cat]"
```

See `pyproject.toml` for all optional dependency groups, including CUDA-specific XGBoost options.

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

This specifies a random forest, an Aspect transformation generating `log_x` from `x`, the model's input features, and the prediction target.

Treeton converts the selected feature columns into a numeric matrix before passing them to the estimator.

A single-model specification is represented internally as an ensemble with one member. Existing configurations remain supported.

### 3. Train

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint model.tt
```

The `model.tt` directory contains the fitted estimator, preprocessing pipeline and checkpoint metadata.

### 4. Predict

```bash
treeton predict model.tt \
    --trust-sklearn-trees \
    --data unseen.csv \
    --output predictions.parquet
```

The output contains the input observations and a prediction column named `prediction:y`.

The `--trust-sklearn-trees` flag explicitly permits loading sklearn's internal tree objects from a checkpoint you trust. See [Checkpoint security](#checkpoint-security).

You can also save predictions as a Providence dataset checkpoint:

```bash
treeton predict model.tt \
    --trust-sklearn-trees \
    --data unseen.csv \
    --checkpoint predictions.pv
```

Both output options can be used together.

### 5. Train an ensemble

To train five independently seeded random forests, replace the `box.model` section of `experiment.json` with:

```json
{
  "kind": "ensemble",
  "schema_version": 1,
  "base": {
    "class_path": "sklearn.ensemble.RandomForestRegressor",
    "init_args": {
      "n_estimators": 100
    }
  },
  "n_members": 5,
  "seed": 42
}
```

Then run:

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint ensemble.tt
```

The ensemble produces a mean prediction by default. Individual member predictions are also accessible through the Python API.

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
| `--model`, `-m` | Bundled model configuration alias |
| `--pipeline`, `-p` | Override the configured Aspect pipeline |
| `--set` | Override a configuration value |
| `--cache` | Directory for data and preprocessing caches |

### Configuration overrides

Use `--set` to override nested configuration values without editing the JSON file:

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint model.tt \
    --set box.model.init_args.n_estimators=500 \
    --set box.model.init_args.max_depth=12
```

For an ensemble configuration:

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint ensemble.tt \
    --set box.model.n_members=10 \
    --set box.model.base.init_args.n_estimators=300
```

Overrides use `dotted.path=value` notation. Values are parsed as JSON where possible, supporting integers, floats, booleans, `null`, arrays and objects.

For example:

```bash
--set box.features='["log_x","another_feature"]'
```

The corresponding feature must exist in the input data or be generated by the configured pipeline.

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
| `--output`, `-o` | Output table, such as CSV or Parquet |
| `--checkpoint`, `-c` | Output Providence dataset checkpoint |
| `--trust-sklearn-trees` | Trust sklearn tree objects during skops deserialization |
| `--start` | First row to predict, inclusive |
| `--stop` | Row at which to stop, exclusive |
| `--pipeline`, `-p` | Override preprocessing pipeline |
| `--config` | Optional JSON prediction settings |
| `--set` | Override prediction settings |
| `--cache` | Cache directory |

For example, predict a subset of a larger dataset:

```bash
treeton predict model.tt \
    --trust-sklearn-trees \
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

An experiment configuration has a small number of top-level sections:

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

The model specification supports single estimators and declarative ensembles.

#### Single estimator

Specify a fully qualified Python class path and constructor arguments:

```json
{
  "class_path": "sklearn.ensemble.ExtraTreesRegressor",
  "init_args": {
    "n_estimators": 200,
    "random_state": 42
  }
}
```

This format remains backward compatible and is internally represented as a one-member `Ensemble`.

Other sklearn-compatible estimators can be instantiated in the same way, although fitting options and serialization support may vary.

#### Homogeneous ensemble

Use `kind: "ensemble"` with `base` and `n_members`:

```json
{
  "kind": "ensemble",
  "schema_version": 1,
  "base": {
    "class_path": "xgboost.XGBRegressor",
    "init_args": {
      "n_estimators": 300,
      "max_depth": 6
    }
  },
  "n_members": 5,
  "seed": 42
}
```

Treeton clones the estimator and assigns successive seeds (`42`, `43`, `44`, `45`, `46`), provided the estimator exposes a `random_state` or `random_seed` parameter.

The members are fitted independently against the same training data and feature representation.

Different seeds do not necessarily guarantee meaningful model diversity; this depends on the estimator and its stochastic training settings.

#### Heterogeneous ensemble

Use `members` to define a named collection of estimators:

```json
{
  "kind": "ensemble",
  "schema_version": 1,
  "members": {
    "rf": {
      "class_path": "sklearn.ensemble.RandomForestRegressor",
      "init_args": {
        "n_estimators": 500,
        "random_state": 42
      }
    },
    "xgb": {
      "class_path": "xgboost.XGBRegressor",
      "init_args": {
        "n_estimators": 300,
        "max_depth": 6,
        "random_state": 43
      }
    }
  },
  "weights": [1, 2]
}
```

Models receive the same input feature matrix and must produce compatible prediction shapes.

By default, predictions are averaged with equal weights. Optional positive weights can be supplied in the order of the configured members.

The example above gives XGBoost twice the weight of the random forest.

Individual models retain their backend-specific fitting and serialization behaviour.

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

The `features` list refers to logical feature names, including transformed output names.

A model can use both existing and transformed columns:

```json
{
  "features": ["numeric_feature", "log_x"]
}
```

All members of an ensemble currently share the same Aspect pipeline.

### `box.features` and `box.target`

Features are supplied as an ordered list of column names. Each selected column can contain a scalar or a fixed-length numeric vector; Treeton concatenates them into a numeric feature matrix.

The target may be a single column or a list of columns:

```json
{
  "features": ["fingerprint", "descriptors"],
  "target": ["activity_a", "activity_b"]
}
```

Multi-target training requires estimators that support multiple outputs. All members of an ensemble must produce compatible target dimensions.

### `fit` and `save`

The `fit` section passes keyword arguments to `Box.fit()`. The `save` section supplies checkpoint options, including those passed to the Aspect pipeline.

For example:

```json
{
  "save": {
    "save_source_data": true,
    "save_transformed": false
  }
}
```

Whether source data are embedded or referenced affects checkpoint size and portability.

Backend-specific fitting, including validation and early stopping, is delegated to the appropriate adapter.

## Python API

The core abstraction is `treeton.Box`, which combines a model or ensemble, feature definitions and an Aspect pipeline.

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

`Box.predict()` returns a Hugging Face Dataset containing prediction columns alongside the prepared input data.

### Save and restore

```python
box.save("model.tt")

restored = Box.load(
    "model.tt",
    trusted_types=("sklearn.tree._tree.Tree",),
)

predictions = restored.predict("unseen.csv")
```

The model and preprocessing pipeline are restored together.

The `trusted_types` argument is needed for sklearn tree checkpoints serialized through skops. Review untrusted types before authorising them.

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

Support for fitting does not guarantee native or safe checkpoint serialization. Unsupported estimator types may require explicit opt-in to unsafe joblib serialization.

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

Ensembles use their aggregated prediction by default.

### Verify the training data source

When the original data source is available through Providence, its recorded provenance can be accessed and verified:

```python
source = box.training_data_source

if source is not None:
    print(box.verify_training_data_source())
```

A returned `None` indicates that no verifiable source is available.

Source verification does not independently guarantee reproducibility of the entire software environment.

## Ensembles

`treeton.Ensemble` provides a composable estimator interface independent of `Box`.

An ensemble can contain one or more compatible estimators, supplied as a single object, an iterable, or a mapping of named members.

### Repeated estimators

```python
from sklearn.ensemble import RandomForestRegressor
from treeton import Ensemble

model = Ensemble.repeat(
    RandomForestRegressor(n_estimators=300),
    n=5,
    seed=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
member_predictions = model.predict(
    X_test,
    disaggregate=True,
)
```

For a single-target regression problem:

- `predict(X)` returns shape `(n_samples,)`.
- `predict(X, disaggregate=True)` returns shape `(n_samples, n_members, 1)`.

The member-level output preserves both the ensemble-member and target dimensions, including for single-member ensembles.

### Heterogeneous ensembles

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from treeton import Ensemble

model = Ensemble(
    {
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
        ),
        "xgb": XGBRegressor(
            n_estimators=300,
            random_state=43,
        ),
    },
    weights=[1, 2],
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
member_predictions = model.predict_members(X_test)

print(list(model.members))
```

The named members can be accessed through `model.members`.

Each member uses its own backend-specific adapter for fitting and prediction. Aggregation is performed by `Ensemble`.

### One-member ensembles

A single compatible estimator can be wrapped directly:

```python
model = Ensemble(
    RandomForestRegressor(
        n_estimators=300,
        random_state=42,
    )
)
```

A one-member ensemble preserves the conventional estimator prediction API, while exposing the same member-level interface as a larger ensemble.

### Declarative ensembles

```python
config = {
    "kind": "ensemble",
    "schema_version": 1,
    "base": {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {"n_estimators": 300},
    },
    "n_members": 5,
    "seed": 42,
}

model = Ensemble.from_config(config)

assert model.to_config() == config
```

`Ensemble.from_config()` retains a defensive copy of the original declarative specification, including its submodel configurations.

`to_config()` returns that specification rather than attempting to reconstruct it from fitted model state.

Ensembles created directly from Python estimator objects do not necessarily have a reconstructable declarative configuration.

### Standalone ensemble checkpoints

An ensemble can be saved independently of a `Box`:

```python
model = Ensemble.from_config(config)
model.fit(X_train, y_train)

model.save("ensemble/model.json")
```

Restore it using:

```python
restored = Ensemble.load(
    "ensemble/model.json",
    trusted=("sklearn.tree._tree.Tree",),
)

predictions = restored.predict(X_test)

assert restored.to_config() == config
```

The ensemble manifest records member names, weights, codec metadata and the original declarative specification when available.

Each fitted member is saved through its own codec.

The ensemble checkpoint does not contain an Aspect pipeline or training dataset; use `Box.save()` to retain those resources.

### Current limitations

Ensemble aggregation currently uses a weighted arithmetic mean, appropriate for numeric regression outputs.

Further aggregation strategies, classification probability alignment, member-specific feature pipelines and nested ensembles are not yet implemented.

Members are fitted sequentially. Individual estimators can still use their own parallelism or GPU resources.

## Molecular machine learning

Treeton is intended to work with scientific feature representations, including molecular fingerprints and descriptors.

For chemical featurisation