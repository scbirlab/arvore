
# 🌳 Treeton

![GitHub Actions](https://img.shields.io/github/actions/workflow/status/scbirlab/treeton/python-test.yml)
![Python versions](https://img.shields.io/pypi/pyversions/treeton)
![PyPI](https://img.shields.io/pypi/v/treeton)

**Treeton makes it straightforward to train, evaluate, checkpoint and share machine-learning models.**

It provides a common interface for scikit-learn-compatible estimators, including random forests, XGBoost, LightGBM and CatBoost. Models can be combined into ensembles, trained using reproducible configurations and restored with their feature preprocessing and training-data provenance.

Treeton also supports **prediction diagnostics**, such as ensemble disagreement, tree-based uncertainty and distance from previously observed data.

Use it from Python or the command line, including for experiments running on HPC systems.

Treeton is in alpha development and currently focuses on numeric regression.

## Features

- **Train different models consistently.** Use random forests, XGBoost, LightGBM, CatBoost or other compatible estimators.
- **Build ensembles.** Combine repeated models or different estimator types, with optional prediction weights.
- **Save and share checkpoints.** Retain trained models, preprocessing and training-data provenance.
- **Score predictions.** Estimate ensemble disagreement, tree variation and novelty relative to training observations.
- **Run reproducible experiments.** Define models and preprocessing in JSON, with command-line overrides.
- **Work with large datasets.** Use batched prediction and checkpointed datasets.

Treeton uses [Aspect](https://github.com/scbirlab/aspect) for data transformations and [Providence](https://github.com/scbirlab/providence) for data loading and provenance.

## Installation

Python 3.11 or later is required.

```bash
pip install treeton
```

For the latest development version:

```bash
pip install "git+https://github.com/scbirlab/treeton.git@dev"
```

Install optional model backends or chemistry support as needed:

```bash
pip install "treeton[trees]"  # XGBoost and LightGBM
pip install "treeton[cat]"    # CatBoost
pip install "treeton[chem]"   # Molecular fingerprints and descriptors
pip install "treeton[bio]"    # Biological feature tools
```

## Quick start

### 1. Prepare some data

Treeton accepts datasets in formats supported by Providence, including CSV and Parquet.

For a small example:

```bash
printf 'x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n9,19\n10,21\n' > train.csv

printf 'x\n2.5\n5.5\n8.5\n' > candidates.csv
```

### 2. Define an experiment

Create `experiment.json`:

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
  },
  "fit": {
    "skip_eval": true
  }
}
```

The configuration defines which model to train, which columns to use and how the data should be prepared.

### 3. Train and save

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint model.tt
```

The checkpoint stores the fitted model alongside its configuration, preprocessing and training-data information.

### 4. Make predictions

```bash
treeton predict model.tt \
    --trust-sklearn-trees \
    --data candidates.csv \
    --output predictions.csv
```

The output contains the original data and a `prediction:y` column.

`--trust-sklearn-trees` authorises loading scikit-learn tree objects from a checkpoint you trust.

### 5. Add prediction diagnostics

```bash
treeton predict model.tt \
    --trust-sklearn-trees \
    --data candidates.csv \
    --score tree_sd inverse_leaf_occupancy euclidean_distance \
    --output scored.csv
```

Alongside the predictions, this adds columns such as:

- `score:tree_sd:y`
- `score:inverse_leaf_occupancy`
- `score:euclidean_distance`

These help identify predictions made with greater model disagreement or farther from the training observations.

## Prediction scoring

Treeton can calculate multiple uncertainty and novelty diagnostics in the same prediction run.

| Scorer | What it measures |
| --- | --- |
| `ensemble_sd` | Disagreement between ensemble members |
| `tree_sd` | Variation in predictions across individual trees |
| `inverse_leaf_occupancy` | How sparsely populated the candidate's tree leaves are in the training data |
| `euclidean_distance` | Distance to the nearest training observation in model feature space |
| `tanimoto_distance` | Chemical fingerprint distance to the nearest training compound |

Higher values generally indicate greater disagreement or novelty.

These quantities have different scales and interpretations. **They are diagnostics, not automatically calibrated probabilities of error.** Some scorers also require particular model capabilities.

### Python API

```python
from treeton import Box
from treeton.scorers import InverseLeafOccupancy

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

results = box.predict(
    "candidates.csv",
    score=[
        "tree_sd",
        "euclidean_distance",
        InverseLeafOccupancy(
            normalization="reference_percentile"
        )
    ]
)
```

Reference-percentile normalisation expresses leaf occupancy relative to the distribution observed in the training data. It can help compare novelty scores from different models.

To assign custom output names, pass a dictionary:

```python
results = box.predict(
    "candidates.csv",
    score={
        "forest_variation": "tree_sd",
        "feature_distance": "euclidean_distance",
        "leaf_percentile": InverseLeafOccupancy(
            normalization="reference_percentile"
        )
    }
)
```

The resulting score columns use those names.

### Molecular similarity

Treeton can estimate chemical novelty using the Tanimoto distance between binary molecular fingerprints.

```python
from treeton.scorers import TanimotoDistance

results = box.predict(
    "compounds.csv",
    score=TanimotoDistance(
        structure_column="smiles",
        column="fingerprint"
    )
)
```

Existing fingerprints are reused when available. Otherwise, fingerprints can be generated from SMILES using Aspect and Schemist.

The molecular fingerprint need not be one of the model's predictive features.

Install chemistry support with:

```bash
pip install "treeton[chem]"
```

Exact nearest-reference similarity is used by default. Large reference libraries can make this calculation expensive.

### Configured scoring from the CLI

For parameterised scorers, create a prediction configuration such as `predict.json`:

```json
{
  "batch_size": 128,
  "score": {
    "leaf_percentile": {
      "class_path": "treeton.scorers.leaves.InverseLeafOccupancy",
      "init_args": {
        "normalization": "reference_percentile"
      }
    },
    "feature_distance": "euclidean_distance"
  }
}
```

Then run:

```bash
treeton predict model.tt \
    --trust-sklearn-trees \
    --data candidates.csv \
    --config predict.json \
    --score tree_sd \
    --output scored.parquet
```

Scorers from the configuration and command line are combined. Custom scorer classes can also be instantiated from installed packages using `class_path` and `init_args`.

## Ensembles

Treeton supports ensembles of repeated models and mixtures of different estimator types.

### Repeated models

To train an ensemble of five random forests, use this model configuration inside `experiment.json`:

```json
{
  "kind": "ensemble",
  "base": {
    "class_path": "sklearn.ensemble.RandomForestRegressor",
    "init_args": {
      "n_estimators": 200
    }
  },
  "n_members": 5,
  "seed": 42
}
```

Treeton constructs separately seeded members and averages their predictions.

### Mixed models

Different estimators can also be combined:

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from treeton import Ensemble

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

The weights determine each member's contribution to the ensemble prediction.

Ensembles currently share one feature representation across their members.

### Inspect individual predictions

Use `--disaggregate` from the CLI:

```bash
treeton predict ensemble.tt \
    --trust-sklearn-trees \
    --data candidates.csv \
    --score ensemble_sd \
    --disaggregate \
    --output predictions.parquet
```

This produces member-level predictions rather than only the ensemble mean. Scoring remains available independently of the output format.

In Python, use `model.predict_members(X)` or `box.predict(..., disaggregate=True)`.

## Training, evaluation and checkpoints

### Python workflow

```python
from treeton import Box

box = Box.from_config({
    "model": {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {"n_estimators": 200}
    },
    "features": ["x"],
    "target": "y",
    "pipeline": {}
})

box.fit("train.csv", skip_eval=True)
box.save("model.tt")

restored = Box.load(
    "model.tt",
    trusted_types=("sklearn.tree._tree.Tree",)
)

predictions = restored.predict("candidates.csv")
```

You can also evaluate trained models on held-out data:

```python
predicted, metrics = box.evaluate(
    "validation.csv",
    metrics=["rmse", "mae", "pearson", "spearman"]
)
```

### Reproducible checkpoints

A `Box` checkpoint retains the fitted model, model configuration, fitted preprocessing and training-data provenance.

Depending on the data source and checkpoint settings, training data may be stored directly or retained by reference.

Reference-based prediction scorers depend on the original training data remaining accessible.

Treeton uses native model formats when supported and skops for scikit-learn estimators. Loading some sklearn checkpoints requires explicitly trusting their internal tree types.

Only load checkpoints from sources you trust. Unsafe joblib serialization is available as an explicit fallback for otherwise unsupported estimators.

## Advanced usage

### Feature transformations

Treeton delegates preprocessing to Aspect.

For example, a pipeline can create a transformed feature:

```json
{
  "column_transforms": {
    "log_x": ["x", "log"]
  }
}
```

The model can then select `"log_x"` in its feature configuration.

Aspect supports other numerical, categorical, chemical and biological transformations. See the [Aspect documentation](https://github.com/scbirlab/aspect).

### Configuration overrides

CLI options can override experiment configuration without editing JSON files:

```bash
treeton train train.csv \
    --config experiment.json \
    --checkpoint model.tt \
    --set box.model.init_args.n_estimators=500
```

Multiple `--set` arguments are supported, including nested ensemble settings.

### Custom scorers

You can supply your own prediction diagnostic as a callable:

```python
def custom_score(context):
    return context.X[:, 0]

results = box.predict(
    "candidates.csv",
    score={"my_score": custom_score}
)
```

Scorers receive a context containing the model, predictions, candidate data and any shared inference resources they request.

More advanced scorers can implement preparation hooks to access training data, build reference indices or construct calibration resources once before batched prediction.

This supports integration with third-party uncertainty and novelty methods.

## Development

Clone the repository and install development dependencies:

```bash
git clone https://github.com/scbirlab/treeton.git
cd treeton
pip install -e ".[dev]"
```

Run the Python tests:

```bash
pytest -q
pytest --doctest-modules -q treeton
```

Run the end-to-end CLI scorer smoke test:

```bash
bash test/smoke_scorers.sh
```

### Project components

| Component | Responsibility |
| --- | --- |
| `Box` | Training, feature selection, prediction and checkpointing |
| `Ensemble` | Model composition and prediction aggregation |
| Model adapters | Backend-specific fitting, inference and persistence |
| Scorers | Uncertainty and novelty diagnostics |
| Aspect | Feature transformation |
| Providence | Dataset resolution, provenance and checkpointing |

The scoring API supports optional preparation, compatibility checks and arbitrary per-invocation state. Model capabilities such as tree predictions and leaf indices are exposed through the adapter layer.

The current implementation focuses on regression and exact reference-distance calculations. Future work may include approximate nearest-neighbour backends, additional uncertainty scorers and evaluation methods incorporating prediction diagnostics.

## Related projects

- [Providence](https://github.com/scbirlab/providence) — Data loading, provenance and checkpointing.
- [Aspect](https://github.com/scbirlab/aspect) — Feature processing and transformation.
- [Eluent](https://github.com/scbirlab/eluent) — Dataset grouping and splitting.
- [Schemist](https://github.com/scbirlab/schemist) — Molecular featurisation.
- [Vectome](https://github.com/scbirlab/vectome) — Biological feature representations.

## License

MIT. See [LICENSE](https://github.com/scbirlab/treeton/blob/main/LICENSE).
