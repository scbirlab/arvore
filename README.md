# 🌳 Treeton

![GitHub Actions](https://img.shields.io/github/actions/workflow/status/scbirlab/treeton/python-test.yml)
![Python versions](https://img.shields.io/pypi/pyversions/treeton)
![PyPI](https://img.shields.io/pypi/v/treeton)

**Treeton** is a lightweight framework for reproducible machine learning with scikit-learn-compatible estimators, composable ensembles, declarative configuration, data provenance and portable model checkpoints.

It provides a common interface for random forests, XGBoost, LightGBM, CatBoost and other estimators implementing `fit(X, y)` and `predict(X)`.

Treeton integrates with [Providence](https://github.com/scbirlab/providence) for dataset loading and provenance and [Aspect](https://github.com/scbirlab/aspect) for feature transformations.

The goal is to make experiments straightforward to **configure, train, evaluate, checkpoint and share**, while preserving the model, fitted preprocessing and training-data provenance needed for reproducibility.

Treeton is in alpha development. Its current focus is numeric regression.

## Features

- **Estimator-agnostic training:** Use sklearn-compatible models through a common interface.
- **Composable ensembles:** Combine repeated estimators or heterogeneous model backends, with optional prediction weights.
- **Member-level predictions:** Access individual model outputs alongside aggregated predictions.
- **Typed configuration:** Validate model specifications using frozen dataclasses, with canonical and original representations.
- **Unified model adapters:** Backend-specific fitting, prediction, serialization and restoration through one registry.
- **Native serialization:** XGBoost (`.ubj`), LightGBM (`.txt`), CatBoost (`.cbm`) and skops, with explicit opt-in joblib fallback.
- **Composable preprocessing:** Use Aspect to generate, combine and reuse fitted features.
- **Training-data provenance:** Retain data sources and transformations in checkpoints through Providence and Aspect.
- **Batched inference:** Predict across Hugging Face Datasets without materializing the entire prediction matrix at once.
- **Evaluation:** Regression metrics including RMSE, MAE, Pearson's r and Spearman's ρ.
- **Python API and CLI:** Use the same experiment abstractions interactively or on HPC systems.

## Installation

Treeton requires Python 3.11 or later.

```bash
pip install "git+https://github.com/scbirlab/treeton.git@dev"
```

For development:

```bash
git clone https://github.com/scbirlab/treeton.git
cd treeton
pip install -e ".[dev]"
```

Optional backend support:

```bash
pip install "treeton[xgb]"    # XGBoost
pip install "treeton[lgbm]"   # LightGBM
pip install "treeton[cat]"    # CatBoost
pip install "treeton[trees]"  # XGBoost and LightGBM
pip install "treeton[chem]"   # Schemist
pip install "treeton[bio]"    # Vectome
```

For the unreleased refactor branch, replace `@dev` with `@refactor/typed-adapters`.

## Quick start

### 1. Create a dataset

```bash
printf 'x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n9,19\n10,21\n' > train.csv

printf 'x\n2.5\n5.5\n8.5\n' > unseen.csv
```

### 2. Define an experiment

Save this as `experiment.json`:

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
    "features": ["x"],
    "target": "y"
  }
}
```

The experiment specifies a model, feature columns, prediction target and preprocessing pipeline.

### 3. Train and checkpoint

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint model.tt
```

The checkpoint includes the fitted model, its configuration, fitted Aspect preprocessing and training-data information.

### 4. Predict

```bash
treeton predict model.tt \
    --trust-sklearn-trees \
    --data unseen.csv \
    --output predictions.parquet
```

Predictions are appended as columns such as `prediction:y`.

The trust flag authorizes loading sklearn's internal tree representation from a trusted skops checkpoint.

### 5. Train an ensemble

Replace `box.model` in the experiment configuration with:

```json
{
  "kind": "ensemble",
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

This constructs five independently seeded random forests, fitted against the same data and feature representation.

By default, their predictions are averaged.

## Configuration

Treeton supports single-estimator specifications and homogeneous or heterogeneous ensembles.

Configuration objects are validated using frozen dataclasses in `treeton.config`.

### Single estimators

```json
{
  "class_path": "sklearn.ensemble.RandomForestRegressor",
  "init_args": {
    "n_estimators": 200,
    "random_state": 42
  }
}
```

`class_path` identifies an importable Python class. `init_args` contains constructor arguments.

A single-estimator specification is internally represented as a one-member ensemble when loaded through `Ensemble.from_config()`.

### Homogeneous ensembles

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

The base estimator is cloned into five distinct instances.

When supported by the estimator, Treeton assigns successive `random_state` or `random_seed` values.

Set `seed` to `null` to avoid assigning member seeds. This is useful for estimators without an appropriate random-seed parameter.

### Heterogeneous ensembles

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

Members are named and fitted independently.

Predictions are combined using a weighted arithmetic mean. In this example, XGBoost contributes twice the weight of the random forest.

All members currently share the same feature representation and must produce compatible prediction shapes.

### Typed configuration objects

Treeton exposes four core configuration and artifact types:

| Type | Responsibility |
| --- | --- |
| `EstimatorConfig` | Estimator class and constructor arguments |
| `EnsembleConfig` | Homogeneous or heterogeneous ensemble specification |
| `ModelArtifact` | Model serialization metadata and integrity information |
| `EnsembleManifest` | Ensemble checkpoint metadata and member definitions |

For example:

```python
from treeton.config import EnsembleConfig

config = EnsembleConfig.from_dict({
    "kind": "ensemble",
    "base": {
        "class_path": "sklearn.ensemble.RandomForestRegressor"
    },
    "n_members": 3
})

print(config.n_members)
print(config.seed)
```

Configuration objects are frozen and nested configuration data are stored in immutable mappings.

They provide two serialization representations:

```python
canonical = config.to_dict()
authored = config.to_dict(raw=True)
```

**Canonical representation** includes resolved Treeton configuration defaults and is the default serialization format.

**Authored representation** preserves the original specification, including fields the user omitted.

This distinction supports both reproducibility and preservation of the user's original intent.

Canonicalization currently applies to Treeton's configuration schema. Estimator-specific constructor defaults that have not been explicitly supplied are not universally expanded, so reproducibility across estimator-library versions also depends on the software environment.

### Configuration validation

Typed configurations validate:

- Estimator class paths and constructor argument structures.
- Ensemble schema versions.
- Mutually exclusive `base` and `members` definitions.
- Positive member counts and valid seeds.
- Positive, finite ensemble weights.
- Checkpoint metadata and artifact checksums.
- Unique member names and unsupported nested ensembles.

Validation is performed before model construction or checkpoint restoration where applicable.

### Configuration overrides

The CLI supports dotted-path overrides:

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint model.tt \
    --set box.model.init_args.n_estimators=500 \
    --set box.model.init_args.max_depth=12
```

For a homogeneous ensemble:

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint ensemble.tt \
    --set box.model.n_members=10 \
    --set box.model.base.init_args.n_estimators=300
```

Values use JSON parsing where possible, supporting integers, floats, booleans, arrays, objects and `null`.

## Python API

### Train and predict

```python
from treeton import Box

box = Box.from_config({
    "model": {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {
            "n_estimators": 200,
            "random_state": 42
        }
    },
    "features": ["x"],
    "target": "y",
    "pipeline": {}
})

box.fit("train.csv", skip_eval=True)

predictions = box.predict("unseen.csv")
print(predictions["prediction:y"])
```

`Box.predict()` returns a Hugging Face Dataset containing prepared observations and prediction columns.

### Evaluate

```python
predicted, scores = box.evaluate(
    "train.csv",
    metrics=["rmse", "mae", "pearson", "spearman"]
)

print(scores)
```

Evaluation requires data containing the target columns. Use held-out data for meaningful performance assessment.

### Save and restore

```python
box.save("model.tt")

restored = Box.load(
    "model.tt",
    trusted_types=("sklearn.tree._tree.Tree",)
)

predictions = restored.predict("unseen.csv")
```

The checkpoint restores both the model and its fitted preprocessing pipeline.

### Use existing estimators

Treeton also accepts existing sklearn-compatible estimator instances:

```python
from sklearn.ensemble import ExtraTreesRegressor
from treeton import Box

box = Box(
    model=ExtraTreesRegressor(
        n_estimators=300,
        random_state=42
    ),
    features=["x"],
    target="y"
)
```

Models supplied directly do not necessarily have a reconstructable declarative configuration. They can still be fitted and checkpointed when a suitable serialization adapter is available.

## Ensembles

`Ensemble` is a standalone model abstraction that implements fitting, prediction, member-level prediction and checkpointing.

### Repeated estimators

```python
from sklearn.ensemble import RandomForestRegressor
from treeton import Ensemble

model = Ensemble.repeat(
    RandomForestRegressor(n_estimators=300),
    n=5,
    seed=42
)

model.fit(X_train, y_train)

mean_prediction = model.predict(X_test)
member_predictions = model.predict_members(X_test)
```

For single-target regression:

- Aggregated prediction shape: `(n_samples,)`
- Member prediction shape: `(n_samples, n_members, 1)`

The same member-level output is available through:

```python
member_predictions = model.predict(
    X_test,
    disaggregate=True
)
```

### Mixed-backend ensembles

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from treeton import Ensemble

model = Ensemble(
    {
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),
        "xgb": XGBRegressor(
            n_estimators=300,
            random_state=43
        )
    },
    weights=[1, 2]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
members = model.predict_members(X_test)
```

Each estimator is handled through the appropriate backend adapter.

### Declarative ensembles

```python
from treeton import Ensemble

spec = {
    "kind": "ensemble",
    "base": {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {"n_estimators": 100}
    },
    "n_members": 3,
    "seed": 42
}

model = Ensemble.from_config(spec)

canonical = model.to_config()
```

The ensemble owns its declarative model configuration.

Configuration is kept separately from fitted estimator state, allowing model specifications and checkpoints to be inspected independently.

### Standalone checkpoints

```python
model.fit(X_train, y_train)
model.save("ensemble/model.json")
```

Restore:

```python
restored = Ensemble.load(
    "ensemble/model.json",
    trusted=("sklearn.tree._tree.Tree",)
)

predictions = restored.predict(X_test)
```

Standalone ensemble checkpoints save each member and a manifest recording member metadata, weights and model configuration.

They do not contain an Aspect pipeline or training data. Use `Box.save()` to preserve those resources.

## Model adapters

Treeton uses registered model adapters to unify training, prediction, serialization and restoration.

Adapters encapsulate backend-specific behavior, including validation sets, early stopping and persistence formats.

| Backend | Persistence format | Restored model |
| --- | --- | --- |
| scikit-learn | `model.skops` | Trainable estimator |
| XGBoost | `model.ubj` | Trainable regressor |
| LightGBM | `model.txt` | Inference-only predictor |
| CatBoost | `model.cbm` | Trainable regressor |
| Arbitrary compatible estimator | `model.joblib` | Unsafe fallback, explicit opt-in |
| Ensemble | `model.json` | Composite of restored members |

Backend dependencies are imported lazily, allowing optional libraries to remain optional.

Fitting and prediction are available for arbitrary compatible estimator types, while native serialization support is currently narrower.

### Adapter registry

Adapters are registered through `register_adapter` and exposed through a common registry.

Internal persistence operations use:

```python
from treeton.models.adapters import save_model, load_model

metadata = save_model(model, "checkpoint/")

restored = load_model(
    "checkpoint/",
    metadata,
    trusted_types=("sklearn.tree._tree.Tree",)
)
```

`save_model()` produces validated artifact metadata containing the adapter identifier, filename, class path, backend version, checksum and trainability.

`load_model()` uses this metadata to select the adapter and verify the artifact before deserialization.

These functions primarily support Box and Ensemble checkpointing. For normal usage, prefer `Box.save()`/`Box.load()` or `Ensemble.save()`/`Ensemble.load()`.

## Checkpoints and reproducibility

A Box checkpoint typically contains:

```text
model.tt/
├── config.json
├── model.json
├── members/
│   ├── 0000/
│   │   └── model.skops
│   └── 0001/
│       └── model.ubj
└── training-data.as/
```

The actual member files depend on the estimator backends.

The checkpoint records:

- Model and ensemble configuration.
- Backend identifiers and serialization formats.
- Backend version metadata.
- SHA-256 checksums for serialized model artifacts.
- Fitted Aspect preprocessing.
- Training-data provenance and retained data as configured.

### Provenance

Providence records source information for supported datasets. Aspect retains fitted preprocessing transformations.

When available, the original data source can be inspected or verified:

```python
source = box.training_data_source

if source is not None:
    print(box.verify_training_data_source())
```

The extent of checkpoint portability depends on whether underlying datasets are embedded or referenced, and whether the original sources remain accessible.

### Checkpoint security

Treeton prefers native model formats or skops for supported estimators.

Skops checks untrusted Python types during model restoration:

```python
box = Box.load(
    "model.tt",
    trusted_types=("sklearn.tree._tree.Tree",)
)
```

Trust only types from checkpoints whose provenance you understand.

Arbitrary estimator objects can be saved with unsafe joblib serialization when explicitly requested:

```python
box.save("custom.tt", allow_unsafe=True)

restored = Box.load(
    "custom.tt",
    allow_unsafe=True
)
```

Joblib relies on pickle and may execute arbitrary code during loading. Never deserialize an untrusted joblib checkpoint.

Serialized artifacts are checked against recorded SHA-256 digests before loading. Checksums detect accidental corruption but do not establish authenticity when the artifact and manifest can both be modified by an attacker.

### Reproducibility limitations

A saved fitted model preserves its learned state, while declarative configuration records how a fresh model should be constructed.

Reproducing training also depends on library versions, numerical backend settings, random seeds, hardware and the availability of the original dataset.

Native LightGBM restoration currently produces an inference-only model.

## Feature processing

Treeton delegates feature transformations to [Aspect](https://github.com/scbirlab/aspect).

A pipeline can transform source columns into model features:

```json
{
  "column_transforms": {
    "log_x": ["x", "log"]
  }
}
```

Feature selection then refers to the transformed columns:

```json
{
  "features": ["log_x"],
  "target": "y"
}
```

Selected scalar and fixed-length vector columns are assembled into a numeric feature matrix.

All ensemble members currently receive the same transformed features.

For molecular machine learning, Aspect can be combined with compatible chemical featurisation tools such as [Schemist](https://github.com/scbirlab/schemist).

## Command-line interface

Treeton provides:

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

Common options include `--validation`, `--model`, `--pipeline`, `--cache` and `--set`.

### Prediction

```bash
treeton predict model.tt \
    --data unseen.csv \
    --output predictions.parquet
```

Prediction supports batched execution and row selection:

```bash
treeton predict model.tt \
    --data large.parquet \
    --start 10000 \
    --stop 20000 \
    --output predictions.parquet
```

The row interval uses Python slicing semantics: `start` is inclusive and `stop` exclusive.

Predictions can also be written to Providence checkpoints with `--checkpoint`.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
pytest --doctest-modules -q treeton
```

The test suite covers configuration validation, model fitting, ensemble behavior, native serialization, checkpoint integrity and API round-tripping.

### Project structure

```text
treeton/
├── box/
│   └── base.py
├── models/
│   ├── adapters.py
│   └── ensemble.py
├── config/
│   ├── configs.py
│   └── instantiate.py
├── metrics/
└── cli/
```

Responsibilities:

- `Box`: preprocessing, feature selection, training, evaluation and checkpoint orchestration.
- `Ensemble`: model composition, aggregation and declarative model configuration.
- `ModelAdapter`: backend-specific fitting, prediction and persistence.
- `treeton.config`: typed configurations, artifact manifests, object instantiation and CLI overrides.
- Providence and Aspect: data resolution, transformations and provenance.

## Roadmap

Planned work includes:

- Composable prediction scorers for ensemble disagreement and tree-based uncertainty.
- Feature-space novelty and reference-based diagnostics.
- Optional integration with mature uncertainty-estimation packages.
- Better interoperability and sharing of model checkpoints.
- Evaluation workflows that can use prediction diagnostics.

These capabilities are planned and are not yet part of the current prediction API.

## Related packages

- [Providence](https://github.com/scbirlab/providence) — Dataset loading, provenance and checkpointing.
- [Aspect](https://github.com/scbirlab/aspect) — Feature transformation and data pipelines.
- [Eluent](https://github.com/scbirlab/eluent) — Dataset splitting and grouping.
- [Schemist](https://github.com/scbirlab/schemist) — Molecular featurisation.
- [Vectome](https://github.com/scbirlab/vectome) — Biological sequence and taxonomy features.

## License

MIT. See [LICENSE](https://github.com/scbirlab/treeton/blob/main/LICENSE).