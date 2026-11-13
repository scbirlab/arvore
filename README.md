# 🌳 Arvore

![GitHub Actions](https://img.shields.io/github/actions/workflow/status/scbirlab/arvore/python-test.yml)
![Python versions](https://img.shields.io/pypi/pyversions/arvore)
![PyPI](https://img.shields.io/pypi/v/arvore)

**Arvore makes tree-based machine learning simple, reproducible and shareable.**

Train random forests, XGBoost, LightGBM, CatBoost and other scikit-learn-compatible estimators through a consistent interface. Combine models into ensembles, evaluate predictions, measure uncertainty and novelty, and share complete experiments as portable checkpoints.

Arvore brings together model training, feature preprocessing, data provenance and prediction diagnostics without imposing a particular modelling workflow. Its composable scoring interface makes it straightforward to compare uncertainty measures or integrate methods from other packages.

Use it interactively from Python or through a reproducible command-line interface, from a laptop to an HPC cluster.

Arvore currently focuses on regression and is under active development.

## Installation

Python 3.11 or later is required.

```bash
pip install arvore
```

Install optional model backends as needed:

```bash
pip install "arvore[trees]"  # XGBoost and LightGBM
pip install "arvore[cat]"    # CatBoost
pip install "arvore[chem]"   # Molecular fingerprints and descriptors
pip install "arvore[bio]"    # Biological feature tools
```

For the latest development version:

```bash
pip install "git+https://github.com/scbirlab/arvore.git@dev"
```

## Quick start

### Command-line interface

Train and save a model using a simple JSON configuration.

Create `experiment.json`:

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
    "features": ["x1", "x2"],
    "target": "y",
    "pipeline": {}
  },
  "fit": {
    "skip_eval": true
  }
}
```

Train a model from a CSV file:

```bash
arvore train train.csv \
    --config experiment.json \
    --checkpoint model.ao
```

Make predictions on new data:

```bash
arvore predict model.ao \
    --trust-sklearn-trees \
    --data candidates.csv \
    --output predictions.csv
```

Add uncertainty and novelty diagnostics with one argument:

```bash
arvore predict model.ao \
    --trust-sklearn-trees \
    --data candidates.csv \
    --score tree_sd feature_distance \
    --output scored.csv
```

The resulting table retains the input columns and adds predictions and requested scores:

| x1 | x2 | prediction:y | score:tree_sd:y | score:feature_distance |
| --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... |

Evaluate against labelled data:

```bash
arvore evaluate model.ao \
    --trust-sklearn-trees \
    --data test.csv \
    --metrics rmse mae pearson spearman \
    --result metrics.json \
    --output evaluated.csv
```

Metrics are printed as JSON, or written to `--result`. `--output` saves the predictions alongside the labelled data.

The same configuration can be reused across datasets and overridden from the command line.

### Python API

The equivalent workflow takes only a few lines:

```python
from arvore import Box

box = Box.from_config({
    "model": {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {
            "n_estimators": 200,
            "random_state": 42
        }
    },
    "features": ["x1", "x2"],
    "target": "y",
    "pipeline": {}
})

box.fit("train.csv", skip_eval=True)

results = box.predict(
    "candidates.csv",
    score=["tree_sd", "feature_distance"]
)

box.save("model.ao")
```

Restore the complete fitted experiment elsewhere:

```python
from arvore import Box

box = Box.load(
    "model.ao",
    trusted_types=("sklearn.tree._tree.Tree",)
)

predictions = box.predict("new_data.csv")
```

The checkpoint preserves the trained estimator, its configuration, fitted preprocessing and training-data provenance. Prepared reference resources for uncertainty and novelty scoring can also be checkpointed and reused.

## Models and ensembles

Arvore supports scikit-learn-compatible estimators through a common training, prediction and persistence interface.

Supported backends include:

- Random forests and other scikit-learn estimators
- XGBoost
- LightGBM
- CatBoost
- Homogeneous and heterogeneous ensembles

Additional estimators can be integrated through model adapters.

### Ensembles

Combine estimators directly through the Python API:

```python
from arvore import Ensemble

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model = Ensemble(
    {
        "rf": RandomForestRegressor(
            n_estimators=200,
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

Arvore also supports ensembles of repeated estimators defined through JSON configurations.

Individual member predictions can be inspected alongside aggregate predictions, including through the CLI's `--disaggregate` option.

## Prediction scoring

Arvore supports composable prediction diagnostics through the `score` argument of `Box.predict()`.

Scores can quantify model disagreement, variability between trees, or distance from observations in the training dataset.

| Scorer | Description |
| --- | --- |
| `ensemble_sd` | Standard deviation of predictions across ensemble members |
| `tree_sd` | Variation in predictions across individual trees |
| `inverse_leaf_occupancy` | Sparsity of training observations in the leaves reached by a candidate |
| `feature_distance` | Distance to nearest training observations in transformed feature space |
| `tanimoto_distance` | Distance to training compounds using molecular fingerprints |

Scores are diagnostics with different scales and interpretations. They should not automatically be interpreted as calibrated prediction errors.

### Composing scorers

Request several scores in the same prediction:

```python
results = box.predict(
    "candidates.csv",
    score=[
        "tree_sd",
        "feature_distance",
        "inverse_leaf_occupancy"
    ]
)
```

Scorers can also be configured as objects:

```python
from arvore.scorers import (
    FeatureDistance,
    InverseLeafOccupancy
)

results = box.predict(
    "candidates.csv",
    score={
        "nearest": FeatureDistance(n_neighbors=1),
        "neighbourhood": FeatureDistance(n_neighbors=10),
        "leaf_percentile": InverseLeafOccupancy(
            normalization="reference_percentile"
        )
    }
)
```

Using a dictionary assigns each scorer a custom output name, allowing different configurations of the same method to be evaluated together.

### Molecular novelty

Chemical novelty can be evaluated independently of the features used by a predictive model.

```python
from arvore.scorers import TanimotoDistance

results = box.predict(
    "compounds.csv",
    score=TanimotoDistance(
        structure_column="smiles",
        column="fingerprint"
    )
)
```

Arvore can reuse existing molecular fingerprints or generate them from molecular structures through its preprocessing integrations.

### External scoring methods

**Any callable accepting an Arvore `ScoreContext` can be used as a scorer.** This allows existing uncertainty estimators, distance metrics or custom diagnostics to be integrated without modifying Arvore.

For example, to use a function from another package:

```python
import numpy as np

from external_package import novelty_score

def external_novelty(context):
    return np.asarray(
        novelty_score(context.X)
    )

results = box.predict(
    "candidates.csv",
    score={"novelty": external_novelty}
)
```

The callable receives the transformed feature matrix and can access the fitted model, predictions, candidate data and other available context.

For methods requiring training-set references, calibration data or additional model capabilities, Arvore provides a more structured scorer interface described under [Extending Arvore](#extending-arvore).

### Reference-based scoring

Some diagnostics require information from the fitted training dataset, such as nearest-neighbour indices, molecular fingerprints or tree-leaf occupancy distributions.

Arvore prepares these resources lazily and caches them on the fitted `Box`. Repeated scoring requests reuse compatible references rather than recomputing them.

```python
box.predict(
    candidates,
    score="feature_distance"
)

# Reuses the same prepared training reference.
box.predict(
    more_candidates,
    score="feature_distance"
)
```

Prepared references can be retained in checkpoints:

```python
box.save(
    "model.ao",
    score=[
        "feature_distance",
        "inverse_leaf_occupancy",
        "tanimoto_distance"
    ]
)
```

References are identified by deterministic specifications, allowing compatible scorers and repeated predictions to share resources.

## Data and preprocessing

Arvore works with datasets resolved through [Providence](https://github.com/scbirlab/providence), including CSV, Parquet and Hugging Face datasets.

Feature transformations are managed through [Aspect](https://github.com/scbirlab/aspect). Preprocessing is fitted alongside the estimator and retained in its checkpoint.

A `Box` specifies the input features, prediction targets, model and optional transformation pipeline.

For example, a pipeline can construct a log-transformed feature:

```json
{
  "column_transforms": {
    "log_x": ["x", "log"]
  }
}
```

The resulting feature can then be selected for model training.

Arvore also supports batched predictions, making it suitable for evaluating large candidate datasets without materialising every prediction batch simultaneously.

Predictions and evaluations run through Hugging Face `Dataset.map` and are fingerprinted by the input data, the requested scorers and a content fingerprint of the fitted `Box` (`box.fingerprint`). For file-backed datasets, repeating a prediction with the same model, data and scorers loads the cached result, including after the checkpoint is reloaded.

## Evaluation and reproducibility

Evaluate models against held-out data using standard regression metrics:

```python
predictions, metrics = box.evaluate(
    "validation.csv",
    metrics=[
        "rmse",
        "mae",
        "pearson",
        "spearman"
    ]
)
```

Metrics follow the scikit-learn argument order, `metric(y_true, y_pred)`, so custom metrics can be passed directly:

```python
from sklearn.metrics import r2_score

predictions, metrics = box.evaluate(
    "validation.csv",
    metrics={"rmse": "rmse", "r2": r2_score}
)
```

### Portable checkpoints

A saved `Box` retains the information needed to recover a fitted experiment:

- Model configuration and fitted estimator
- Feature definitions and fitted preprocessing
- Training-data provenance and retained training references
- Prepared uncertainty and novelty resources, when requested

Training data may be embedded or retained by reference, depending on its source and the checkpoint settings.

Model persistence uses backend-native formats where supported and skops for scikit-learn models. Some scikit-learn checkpoints require explicitly trusting internal estimator types during loading. An unsafe joblib fallback is available for otherwise unsupported estimators.

Only load checkpoints from trusted sources.

### Reproducible command-line experiments

Configurations can be overridden without editing the original experiment definition:

```bash
arvore train train.csv \
    --config experiment.json \
    --checkpoint experiment.ao \
    --set box.model.init_args.n_estimators=500
```

This makes the same experiment definition reusable across model hyperparameters, datasets and automated workflows.

## Extending Arvore

Arvore is designed to accommodate external estimators and prediction diagnostics with minimal integration code.

### Box

`Box` is the central interface for a fitted experiment.

It brings together:

| Component | Responsibility |
| --- | --- |
| Model | A fitted estimator or ensemble |
| Features and targets | Definitions of model inputs and outputs |
| Pipeline | Feature transformations and preprocessing |
| Data provenance | Resolution and retention of training data |
| Scorers | Optional uncertainty and novelty diagnostics |
| References | Reusable training-dependent resources |
| Checkpoint | Persistence of the complete fitted experiment |

The main operations are `fit()`, `predict()`, `evaluate()`, `save()` and `load()`.

Model-specific operations are delegated to adapters, keeping `Box` independent of a particular estimation library.

### Writing a custom Scorer

A minimal scorer only needs to be callable:

```python
import numpy as np

def feature_norm(context):
    return np.linalg.norm(context.X, axis=1)

results = box.predict(
    candidates,
    score={"feature_norm": feature_norm}
)
```

The `ScoreContext` provides backend-independent access to information such as:

- `X`: transformed candidate features
- `predictions`: model predictions
- `model`: fitted model or ensemble
- `batch`: original candidate columns
- `targets`: prediction target names
- `member_predictions`: optional ensemble-member predictions
- `tree_predictions`: optional individual-tree predictions
- `state`: scorer-specific prepared resources
- `box`: the fitted `Box`

More advanced methods can declare requirements and implement preparation hooks.

For example, a scorer wrapping an external uncertainty implementation:

```python
from dataclasses import dataclass
from typing import ClassVar

from arvore.scorers import (
    Requirements,
    ScoreContext,
    register_scorer
)

from external_package import ExternalUncertainty


@register_scorer
@dataclass(frozen=True)
class ExternalScore:
    name: ClassVar[str] = "external_score"

    @property
    def requirements(self):
        return Requirements(
            outputs=("prediction",)
        )

    def prepare(self, box, *, batch_size=1024):
        # Construct or retrieve any resources needed
        # by the external implementation.
        return ExternalUncertainty(box.model)

    def __call__(self, context: ScoreContext):
        return context.state.score(context.X)
```

The scorer can then be invoked by name:

```python
results = box.predict(
    candidates,
    score="external_score"
)
```

The external implementation above is illustrative; its constructor and `score()` method should be adapted to the actual library.

### Scorer extension points

Scorers can implement several optional hooks:

| Hook | Purpose |
| --- | --- |
| `requirements` | Declare required candidate columns, model capabilities and prediction outputs |
| `prepare_candidates(data, box=...)` | Generate or resolve auxiliary candidate columns |
| `validate(box, prepared)` | Check compatibility before scoring |
| `prepare(box, batch_size=...)` | Initialise training references, indices or calibration resources |
| `__call__(context)` | Evaluate the scorer for a prediction batch |
| `finalize(result, name=..., box=...)` | Inspect the completed result or issue diagnostics |

Arvore validates declared requirements and supplies common inference resources, including ensemble-member predictions and individual-tree predictions, through model adapters where supported.

A scorer can return one value per sample or per target. It may also return a mapping containing a primary score and additional named outputs.

### Reusing reference resources

Scorers that depend on training data can request reusable resources through `box.references.get_or_build()`.

```python
reference = box.references.get_or_build(
    {
        "type": "external_reference",
        "version": 1,
        "metric": "euclidean"
    },
    builder
)
```

Here, `builder` is a callable that constructs the resource when it has not already been prepared.

Arvore hashes the canonical resource specification to obtain a deterministic cache key. Equivalent requests reuse the same prepared resource, including across scorer instances.

Reference types supporting Arvore's checkpoint persistence can be saved and restored alongside the fitted model.

This enables integration with external nearest-neighbour libraries, model-specific uncertainty implementations, density estimators or calibration methods without embedding their algorithms in `Box`.

### Model adapters

Custom estimators can be supported through Arvore's adapter registry.

Adapters handle estimator-specific fitting, prediction, model capabilities and persistence. This allows scorers to request capabilities such as individual-tree predictions without depending directly on XGBoost, LightGBM, CatBoost or scikit-learn internals.

## Development

Clone the repository and install development dependencies:

```bash
git clone https://github.com/scbirlab/arvore.git
cd arvore
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
```

Run module doctests:

```bash
pytest --doctest-modules -q arvore
```

The project is developed using Python 3.11 and later.

### Related projects

Arvore integrates with other packages in the [scbirlab](https://github.com/scbirlab) ecosystem:

- [Providence](https://github.com/scbirlab/providence) — Data resolution, provenance and checkpointing.
- [Aspect](https://github.com/scbirlab/aspect) — Feature transformation and processing.
- [Eluent](https://github.com/scbirlab/eluent) — Dataset grouping and splitting.
- [Schemist](https://github.com/scbirlab/schemist) — Molecular featurisation.
- [Vectome](https://github.com/scbirlab/vectome) — Biological feature representations.

### Issues, problems, suggestions

Please report bugs, feature requests and suggestions through the [GitHub issue tracker](https://github.com/scbirlab/arvore/issues).

### License

MIT. See [LICENSE](https://github.com/scbirlab/arvore/blob/main/LICENSE).
