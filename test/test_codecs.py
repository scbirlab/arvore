"""Model codec round-trip tests."""

import numpy as np
import pytest

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from treeton.models.codecs import (
    save_model,
    load_model,
)


@pytest.fixture
def data():
    X, y = make_regression(
        n_samples=160,
        n_features=5,
        noise=0.2,
        random_state=42,
    )
    return X[:120], y[:120], X[120:], y[120:]


@pytest.mark.parametrize(
    "backend",
    ["sklearn", "xgboost", "lightgbm", 
    # "catboost",
    ],
)
def test_prediction_roundtrip(tmp_path, data, backend):
    X, y, Xv, yv = data

    if backend == "sklearn":
        pytest.importorskip("skops.io")
        model = RandomForestRegressor(
            n_estimators=20, random_state=42
        )
        model.fit(X, y)

    elif backend == "xgboost":
        xgb = pytest.importorskip("xgboost")
        model = xgb.XGBRegressor(
            n_estimators=80,
            learning_rate=0.1,
            eval_metric="rmse",
            early_stopping_rounds=8,
            n_jobs=1,
        )
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)

    elif backend == "lightgbm":
        lgb = pytest.importorskip("lightgbm")
        model = lgb.LGBMRegressor(
            n_estimators=80,
            min_child_samples=5,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(
            X, y,
            eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(8, verbose=False)],
        )

    else:
        cat = pytest.importorskip("catboost")
        model = cat.CatBoostRegressor(
            iterations=80,
            verbose=False,
            thread_count=1,
        )
        model.fit(
            X, y,
            eval_set=(Xv, yv),
            early_stopping_rounds=8,
        )

    expected = model.predict(Xv)
    metadata = save_model(model, tmp_path)
    restored = load_model(tmp_path, metadata)

    actual = restored.predict(Xv)

    np.testing.assert_allclose(
        expected,
        actual,
        rtol=1e-5,
        atol=1e-6,
    )


def test_checksum_rejection(tmp_path, data):
    X, y, *_ = data
    model = RandomForestRegressor(n_estimators=5)
    model.fit(X, y)

    pytest.importorskip("skops.io")
    metadata = save_model(model, tmp_path)

    path = tmp_path / metadata["filename"]
    path.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path, metadata)



class CustomRegressor:
    """Unsupported estimator for testing unsafe serialization."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_unsafe_joblib_is_opt_in(tmp_path, data):
    X, y, *_ = data

    model = CustomRegressor().fit(X, y)

    with pytest.raises(TypeError, match="No supported codec"):
        save_model(model, tmp_path)

    metadata = save_model(
        model, tmp_path, allow_unsafe=True
    )

    assert metadata["codec"] == "joblib"

    with pytest.raises(ValueError, match="allow_unsafe"):
        load_model(tmp_path, metadata)

    restored = load_model(
        tmp_path, metadata, allow_unsafe=True
    )

    np.testing.assert_allclose(
        model.predict(X),
        restored.predict(X),
    )

