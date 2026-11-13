"""Cross-backend leaf-index capability tests."""

import numpy as np
import pytest

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge

from arvore.models.adapters import get_adapter


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2] ** 2
    return X, y


def check_leaves(model, X):
    adapter = get_adapter(model)

    assert "leaf_indices" in adapter.capabilities(model)

    leaves = adapter.leaf_indices(model, X)

    assert leaves.ndim == 2
    assert leaves.shape[0] == len(X)
    assert leaves.shape[1] >= 1
    assert np.issubdtype(leaves.dtype, np.integer)
    assert np.all(leaves >= 0)

    # One-row batches must retain the tree dimension.
    one = adapter.leaf_indices(model, X[:1])
    np.testing.assert_array_equal(one, leaves[:1])

    # Leaf assignments must not depend on batch size.
    chunks = [
        adapter.leaf_indices(model, chunk)
        for chunk in np.array_split(X, 4)
    ]
    np.testing.assert_array_equal(
        np.concatenate(chunks, axis=0),
        leaves,
    )

    return leaves


@pytest.mark.parametrize(
    "cls",
    [RandomForestRegressor, ExtraTreesRegressor],
)
def test_sklearn_forests(cls, regression_data):
    X, y = regression_data

    model = cls(
        n_estimators=13,
        random_state=42,
    ).fit(X, y)

    leaves = check_leaves(model, X)

    assert leaves.shape == (80, 13)
    np.testing.assert_array_equal(
        leaves,
        model.apply(X),
    )


def test_sklearn_multitarget(regression_data):
    X, y = regression_data

    model = RandomForestRegressor(
        n_estimators=7,
        random_state=42,
    ).fit(X, np.column_stack([y, 2 * y]))

    leaves = check_leaves(model, X)
    assert leaves.shape == (80, 7)


def test_unsupported_estimator(regression_data):
    X, y = regression_data

    model = Ridge().fit(X, y)
    adapter = get_adapter(model)

    assert "leaf_indices" not in adapter.capabilities(model)

    with pytest.raises(NotImplementedError):
        adapter.leaf_indices(model, X)


def test_xgboost(regression_data, tmp_path):
    xgb = pytest.importorskip("xgboost")
    X, y = regression_data

    model = xgb.XGBRegressor(
        n_estimators=12,
        max_depth=3,
        n_jobs=1,
        random_state=42,
    ).fit(X, y)

    leaves = check_leaves(model, X)
    assert leaves.shape == (80, 12)

    path = tmp_path / "model.ubj"
    adapter = get_adapter(model)
    adapter.save(model, path)
    restored = adapter.load(path)

    np.testing.assert_array_equal(
        leaves,
        check_leaves(restored, X),
    )


def test_xgboost_early_stopping(regression_data):
    xgb = pytest.importorskip("xgboost")
    X, y = regression_data

    model = xgb.XGBRegressor(
        n_estimators=40,
        max_depth=2,
        learning_rate=0.3,
        early_stopping_rounds=3,
        n_jobs=1,
        random_state=42,
    )

    model.fit(
        X,
        y,
        eval_set=[(X, -y)],
        verbose=False,
    )

    leaves = check_leaves(model, X)

    # One tree per boosting iteration in this configuration.
    assert leaves.shape[1] == model.best_iteration + 1


def test_lightgbm_checkpoint(regression_data, tmp_path):
    lgb = pytest.importorskip("lightgbm")
    X, y = regression_data

    model = lgb.LGBMRegressor(
        n_estimators=12,
        num_leaves=7,
        min_child_samples=3,
        verbosity=-1,
        n_jobs=1,
        random_state=42,
    ).fit(X, y)

    leaves = check_leaves(model, X)

    path = tmp_path / "model.txt"
    adapter = get_adapter(model)
    adapter.save(model, path)

    restored = adapter.load(path)

    # Must dispatch to the LightGBM adapter,
    # even though restoration returns LightGBMPredictor.
    restored_adapter = get_adapter(restored)
    assert restored_adapter.name == "lightgbm"

    np.testing.assert_array_equal(
        leaves,
        check_leaves(restored, X),
    )

    np.testing.assert_allclose(
        model.predict(X),
        restored.predict(X),
    )


def test_catboost_checkpoint(regression_data, tmp_path):
    cb = pytest.importorskip("catboost")
    X, y = regression_data

    model = cb.CatBoostRegressor(
        iterations=12,
        depth=3,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
        random_seed=42,
    ).fit(X, y)

    leaves = check_leaves(model, X)

    path = tmp_path / "model.cbm"
    adapter = get_adapter(model)
    adapter.save(model, path)

    restored = adapter.load(path)

    np.testing.assert_array_equal(
        leaves,
        check_leaves(restored, X),
    )
