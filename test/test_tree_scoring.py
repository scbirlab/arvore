"""Integration tests for within-forest prediction scoring."""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from arvore import Box, Ensemble
from arvore.models.adapters import get_adapter


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




@pytest.mark.parametrize(
    "score",
    [
        None,
        "ensemble_sd",
        "tree_sd",
        ["ensemble_sd", "tree_sd"],
    ],
)
def test_disaggregate_independent_of_scoring(
    regression_data,
    score,
):
    X, y, data = regression_data

    members = {
        name: RandomForestRegressor(
            n_estimators=n,
            random_state=seed,
        ).fit(X, y)
        for name, n, seed in [
            ("a", 11, 1),
            ("b", 17, 2),
        ]
    }

    box = make_box(
        Ensemble(members, weights=(1, 3)),
        data,
    )

    result = box.predict(
        disaggregate=True,
        score=score,
        batch_size=13,
        map_kwargs={"load_from_cache_file": False},
    )

    assert "prediction:y" not in result.column_names

    for name, member in members.items():
        column = f"prediction:y:{name}"

        assert column in result.column_names

        np.testing.assert_allclose(
            result[column],
            member.predict(X),
        )

    if score is not None:
        requested = (
            [score] if isinstance(score, str)
            else score
        )

        for name in requested:
            assert f"score:{name}:y" in result.column_names


def test_scoring_preserves_prediction_columns(
    regression_data,
):
    X, y, data = regression_data

    members = {
        str(i): RandomForestRegressor(
            n_estimators=10,
            random_state=i,
        ).fit(X, y)
        for i in range(2)
    }

    box = make_box(Ensemble(members), data)

    for disaggregate in (False, True):
        baseline = box.predict(
            disaggregate=disaggregate,
            map_kwargs={"load_from_cache_file": False},
        )

        scored = box.predict(
            disaggregate=disaggregate,
            score=["ensemble_sd", "tree_sd"],
            map_kwargs={"load_from_cache_file": False},
        )

        prediction_cols = [
            c for c in baseline.column_names
            if c.startswith("prediction:")
        ]

        scored_prediction_cols = [
            c for c in scored.column_names
            if c.startswith("prediction:")
        ]

        assert prediction_cols == scored_prediction_cols

        for column in prediction_cols:
            np.testing.assert_allclose(
                baseline[column],
                scored[column],
            )


def test_multitarget_disaggregation(regression_data):
    X, y, data = regression_data

    Y = np.column_stack([y, 2 * y])

    members = {
        name: RandomForestRegressor(
            n_estimators=9,
            random_state=seed,
        ).fit(X, Y)
        for name, seed in [("a", 1), ("b", 2)]
    }

    box = make_box(Ensemble(members), data)
    box.target = ("y1", "y2")

    result = box.predict(
        disaggregate=True,
        score=["ensemble_sd", "tree_sd"],
        map_kwargs={"load_from_cache_file": False},
    )

    for name, member in members.items():
        predicted = member.predict(X)

        for index, target in enumerate(box.target):
            np.testing.assert_allclose(
                result[f"prediction:{target}:{name}"],
                predicted[:, index],
            )

    for target in box.target:
        assert f"score:ensemble_sd:{target}" in result.column_names
        assert f"score:tree_sd:{target}" in result.column_names


def test_single_model_disaggregate_noop(regression_data):
    X, y, data = regression_data

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    ).fit(X, y)

    box = make_box(model, data)

    result = box.predict(
        disaggregate=True,
        score="tree_sd",
        map_kwargs={"load_from_cache_file": False},
    )

    assert "prediction:y" in result.column_names
    assert "score:tree_sd:y" in result.column_names
