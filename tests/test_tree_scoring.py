"""Integration tests for within-forest prediction scoring."""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from treeton import Box, Ensemble
from treeton.models.adapters import get_adapter


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(80, 4))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2] ** 2

    data = Dataset.from_dict({
        f"x{i}": X[:, i]
        for i in range(X.shape[1])
    })

    return X, y, data


def make_box(model, data):
    box = Box.__new__(Box)
    box.model = model
    box.features = ("x0", "x1", "x2", "x3")
    box.target = ("y",)

    box._pre_predict = lambda **kwargs: (
        data,
        SimpleNamespace(column_transforms={}),
    )

    return box


def test_single_forest_tree_sd(regression_data):
    X, y, data = regression_data

    model = RandomForestRegressor(
        n_estimators=17,
        random_state=42,
    ).fit(X, y)

    box = make_box(model, data)

    result = box.predict(
        score="tree_sd",
        batch_size=13,
        map_kwargs={"load_from_cache_file": False},
    )

    trees = get_adapter(model).tree_predictions(model, X)

    np.testing.assert_allclose(
        result["prediction:y"],
        model.predict(X),
    )

    np.testing.assert_allclose(
        result["score:tree_sd:y"],
        trees.std(axis=1)[:, 0],
    )


def test_ensemble_tree_sd(regression_data):
    X, y, data = regression_data

    forests = [
        RandomForestRegressor(
            n_estimators=n,
            random_state=seed,
        ).fit(X, y)
        for n, seed in [(11, 1), (23, 2)]
    ]

    ensemble = Ensemble(
        {"a": forests[0], "b": forests[1]},
        weights=(1.0, 3.0),
    )

    box = make_box(ensemble, data)

    result = box.predict(
        score=["ensemble_sd", "tree_sd"],
        batch_size=17,
        map_kwargs={"load_from_cache_file": False},
    )

    trees = [
        get_adapter(m).tree_predictions(m, X)
        for m in forests
    ]

    member_means = np.stack(
        [v.mean(axis=1) for v in trees],
        axis=1,
    )

    mean = np.average(
        member_means,
        axis=1,
        weights=ensemble.weights,
    )

    within = np.stack(
        [v.var(axis=1) for v in trees],
        axis=1,
    )

    expected_tree_sd = np.sqrt(
        np.average(
            within,
            axis=1,
            weights=ensemble.weights,
        )
    )

    expected_ensemble_sd = np.sqrt(
        np.average(
            (member_means - mean[:, None, :]) ** 2,
            axis=1,
            weights=ensemble.weights,
        )
    )

    np.testing.assert_allclose(
        result["prediction:y"],
        mean[:, 0],
    )

    np.testing.assert_allclose(
        result["score:tree_sd:y"],
        expected_tree_sd[:, 0],
    )

    np.testing.assert_allclose(
        result["score:ensemble_sd:y"],
        expected_ensemble_sd[:, 0],
    )


def test_unsupported_heterogeneous_ensemble(regression_data):
    X, y, data = regression_data

    forest = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    ).fit(X, y)

    ridge = Ridge().fit(X, y)

    box = make_box(
        Ensemble({"forest": forest, "ridge": ridge}),
        data,
    )

    with pytest.raises(ValueError, match="unsupported members"):
        box.predict(score="tree_sd")


def test_tree_sd_batch_independence(regression_data):
    X, y, data = regression_data

    model = RandomForestRegressor(
        n_estimators=15,
        random_state=42,
    ).fit(X, y)

    box = make_box(model, data)

    small = box.predict(
        score="tree_sd",
        batch_size=7,
        map_kwargs={"load_from_cache_file": False},
    )

    large = box.predict(
        score="tree_sd",
        batch_size=80,
        map_kwargs={"load_from_cache_file": False},
    )

    np.testing.assert_allclose(
        small["score:tree_sd:y"],
        large["score:tree_sd:y"],
    )
