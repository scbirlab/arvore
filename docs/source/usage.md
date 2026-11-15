# Usage

## Python

A `Box` combines an [aspect](https://github.com/scbirlab/aspect) feature
pipeline with an estimator, or an `Ensemble` of estimators.

```python
from arvore import Box

box = Box.from_config({
    "model": {
        "class_path": "sklearn.ensemble.RandomForestRegressor",
        "init_args": {"n_estimators": 200, "random_state": 42},
    },
    "features": ["x1", "x2"],
    "target": "y",
})

box.fit("train.csv", skip_eval=True)
predictions = box.predict("candidates.csv", score=["tree_sd", "feature_distance"])
predicted, metrics = box.evaluate("test.csv", metrics=["rmse", "pearson"])

box.save("model.ao")
box = Box.load("model.ao", trusted_types=("sklearn.tree._tree.Tree",))
```

## Command line

The same configuration, under a `box` key, drives the command-line interface:

```bash
arvore train train.csv --config experiment.json --checkpoint model.ao
arvore predict model.ao --trust-sklearn-trees --data candidates.csv --output predictions.csv
arvore evaluate model.ao --trust-sklearn-trees --data test.csv --metrics rmse pearson
```

Override any configuration value with `--set box.model.init_args.n_estimators=500`,
or swap the model for a bundled one with `--model`, for example `--model xgboost`.

## Censored and missing targets

```json
{
    "model": {"class_path": "xgboost.XGBRegressor"},
    "features": ["smiles"],
    "target": "pIC50",
    "censor": "qualifier",
    "loss": {"masked": true}
}
```

`censor` names one numeric column per target: negative for left-censored
values, positive for right-censored and zero for exact. A masked `loss` ignores
missing targets. Evaluation metrics use exact, observed targets only.

See the [README](https://github.com/scbirlab/arvore#readme) for ensembles,
scorers, reference resources and extending Arvore.
