"""Tests for tree estimator fit adapters."""

import numpy as np
import pytest

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from treeton.models import (
    SklearnAdapter,
    XGBoostAdapter,
    LightGBMAdapter,
    CatBoostAdapter,
    get_adapter,
)


@pytest.fixture
def data():
    X, y = make_regression(
        n_samples=120,
        n_features=5,
        noise=0.1,
        random_state=42,
    )
    return X[:90], y[:90], X[90:], y[90:]


def test_sklearn(data):
    X, y, Xv, yv = data

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )
    adapter = get_adapter(model)

    assert isinstance(adapter, SklearnAdapter)

    # Validation is ignored by the generic sklearn adapter.
    adapter.fit(
        model, X, y,
        validation=(Xv, yv),
    )

    assert np.isfinite(model.predict(Xv)).all()


def test_xgboost(data):
    xgb = pytest.importorskip("xgboost")
    X, y, Xv, yv = data

    model = xgb.XGBRegressor(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="rmse",
        n_jobs=1,
        random_state=42,
    )
    adapter = get_adapter(model)

    assert isinstance(adapter, XGBoostAdapter)

    adapter.fit(
        model, X, y,
        validation=(Xv, yv),
        early_stopping_rounds=5,
    )

    assert model.best_iteration >= 0
    assert np.isfinite(model.predict(Xv)).all()


def test_lightgbm(data):
    lgb = pytest.importorskip("lightgbm")
    X, y, Xv, yv = data

    model = lgb.LGBMRegressor(
        n_estimators=60,
        learning_rate=0.1,
        min_child_samples=5,
        verbosity=-1,
        n_jobs=1,
        random_state=42,
    )
    adapter = get_adapter(model)

    assert isinstance(adapter, LightGBMAdapter)

    adapter.fit(
        model, X, y,
        validation=(Xv, yv),
        early_stopping_rounds=5,
    )

    assert model.best_iteration_ > 0
    assert np.isfinite(model.predict(Xv)).all()


def test_catboost(data):
    cat = pytest.importorskip("catboost")
    X, y, Xv, yv = data

    model = cat.CatBoostRegressor(
        iterations=60,
        depth=3,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=False,
        thread_count=1,
        random_seed=42,
    )
    adapter = get_adapter(model)

    assert isinstance(adapter, CatBoostAdapter)

    adapter.fit(
        model, X, y,
        validation=(Xv, yv),
        early_stopping_rounds=5,
    )

    assert model.tree_count_ > 0
    assert np.isfinite(model.predict(Xv)).all()


@pytest.mark.parametrize(
    "adapter",
    [XGBoostAdapter(), LightGBMAdapter(), CatBoostAdapter()],
)
def test_early_stopping_requires_validation(adapter):
    with pytest.raises(ValueError, match="validation"):
        adapter.fit(
            None, None, None,
            early_stopping_rounds=5,
        )


def test_sklearn_rejects_early_stopping(data):
    X, y, *_ = data

    model = RandomForestRegressor(n_estimators=5)

    with pytest.raises(ValueError, match="sklearn"):
        SklearnAdapter().fit(
            model, X, y,
            early_stopping_rounds=5,
        )
