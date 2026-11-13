"""Tests for tree-level adapter capabilities."""

import numpy as np
import pytest

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

from arvore.models.adapters import get_adapter


@pytest.mark.parametrize(
    "estimator_class",
    [RandomForestRegressor, ExtraTreesRegressor],
)
@pytest.mark.parametrize("n_targets", [1, 2])
def test_tree_predictions(estimator_class, n_targets):
    rng = np.random.default_rng(42)

    X = rng.normal(size=(40, 5))
    y = rng.normal(size=(40, n_targets))

    if n_targets == 1:
        y = y[:, 0]

    model = estimator_class(
        n_estimators=11,
        random_state=42,
    )
    model.fit(X, y)

    adapter = get_adapter(model)

    assert "tree_predictions" in adapter.capabilities(model)

    predictions = adapter.tree_predictions(model, X)

    assert predictions.shape == (40, 11, n_targets)

    expected = np.asarray(model.predict(X))

    if expected.ndim == 1:
        expected = expected[:, None]

    np.testing.assert_allclose(
        predictions.mean(axis=1),
        expected,
    )


@pytest.mark.parametrize(
    "model",
    [
        Ridge(),
        GradientBoostingRegressor(),
    ],
)
def test_unsupported_models(model):
    rng = np.random.default_rng(42)

    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30)

    model.fit(X, y)

    adapter = get_adapter(model)

    assert "tree_predictions" not in adapter.capabilities(model)

    with pytest.raises(
        NotImplementedError,
        match="unsupported",
    ):
        adapter.tree_predictions(model, X)


def test_unfitted_forest():
    model = RandomForestRegressor(n_estimators=5)
    adapter = get_adapter(model)

    assert "tree_predictions" in adapter.capabilities(model)

    with pytest.raises(Exception):
        adapter.tree_predictions(
            model,
            np.zeros((4, 3)),
        )
