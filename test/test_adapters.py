
"""Regression tests for unified model adapters."""
import os

import numpy as np
import pytest

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from arvore.models.adapters import (
    ADAPTERS,
    get_adapter,
    get_adapter_by_name,
    get_serialization_adapter,
    save_model,
    load_model,
)


class CustomRegressor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


@pytest.fixture
def fitted_rf():
    X, y = make_regression(
        n_samples=64, n_features=4, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=10, random_state=42
    ).fit(X, y)
    return model, X


def test_single_registry():
    assert set(ADAPTERS) == {
        "skops", "joblib", "xgboost",
        "lightgbm", "catboost", "ensemble",
    }

    for name in ADAPTERS:
        adapter = get_adapter_by_name(name)
        assert adapter.name == name
        assert adapter.filename.startswith("model.")


def test_sklearn_dispatch(fitted_rf):
    pytest.importorskip("skops.io")
    model, _ = fitted_rf

    adapter = get_adapter(model)
    serializer = get_serialization_adapter(model)

    assert serializer.name == "skops"
    assert type(adapter) is type(serializer)


def test_sklearn_roundtrip(tmp_path, fitted_rf):
    pytest.importorskip("skops.io")
    model, X = fitted_rf

    metadata = save_model(model, tmp_path)

    assert metadata["adapter"] == "skops"
    assert metadata["filename"] == "model.skops"

    restored = load_model(
        tmp_path,
        metadata,
        trusted_types=("sklearn.tree._tree.Tree",),
    )

    np.testing.assert_allclose(
        model.predict(X),
        restored.predict(X),
    )


def test_inference_only_capability():
    adapter = get_adapter_by_name("lightgbm")
    print(adapter)
    assert adapter.restored_trainable is False

    for name in ("xgboost", "catboost", "skops"):
        assert get_adapter_by_name(name).restored_trainable


def test_checksum_rejection(tmp_path, fitted_rf):
    pytest.importorskip("skops.io")
    model, _ = fitted_rf

    metadata = save_model(model, tmp_path)
    artifact = tmp_path / metadata["filename"]
    artifact.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path, metadata)



def test_joblib_requires_explicit_authorisation(
    tmp_path, fitted_rf
):
    model, X = fitted_rf
    custom = CustomRegressor().fit(X, model.predict(X))

    with pytest.raises(TypeError, match="No supported codec"):
        get_serialization_adapter(custom)

    metadata = save_model(
        custom,
        tmp_path,
        allow_unsafe=True,
    )

    assert metadata["adapter"] == "joblib"
    assert metadata["filename"] == "model.joblib"

    with pytest.raises(ValueError, match="allow_unsafe"):
        load_model(tmp_path, metadata)

    restored = load_model(
        tmp_path,
        metadata,
        allow_unsafe=True,
    )

    np.testing.assert_allclose(
        custom.predict(X),
        restored.predict(X),
    )



def test_native_serialization_preferred_over_joblib():
    xgb = pytest.importorskip("xgboost")
    lgb = pytest.importorskip("lightgbm")

    for model, expected in (
        (xgb.XGBRegressor(), "xgboost"),
        (lgb.LGBMRegressor(), "lightgbm"),
    ):
        assert (
            get_serialization_adapter(
                model, allow_unsafe=True
            ).name
            == expected
        )
