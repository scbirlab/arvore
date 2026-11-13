"""Reference leaf occupancy tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)

from arvore import Box, Ensemble
from arvore.models.adapters import get_adapter
from arvore.scorers.leaves import (
    LeafReferenceBuilder,
)


def test_sparse_leaf_reference():
    builder = LeafReferenceBuilder()

    builder.update(np.array([
        [0, 10],
        [0, 20],
    ]))

    builder.update(np.array([
        [1, 20],
    ]))

    reference = builder.finish()

    assert reference.n_reference == 3

    query = np.array([
        [0, 20],
        [1, 10],
        [2, 99],
    ])

    scores = reference.inverse_score(query)

    expected = [
        1 / 3,
        (1 / 2 + 1 / 2) / 2,
        1.0,
    ]

    np.testing.assert_allclose(scores, expected)


@pytest.fixture
def datasets_and_models():
    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(48, 4))
    y_train = (
        X_train[:, 0]
        - 0.5 * X_train[:, 1]
        + X_train[:, 2] ** 2
    )
    X_test = rng.normal(size=(17, 4))

    def dataset(X):
        return Dataset.from_dict({
            f"x{i}": X[:, i]
            for i in range(X.shape[1])
        })

    models = {
        "rf": RandomForestRegressor(
            n_estimators=11,
            random_state=42,
        ).fit(X_train, y_train),
        "extra": ExtraTreesRegressor(
            n_estimators=17,
            random_state=43,
        ).fit(X_train, y_train),
    }

    return (
        X_train,
        X_test,
        dataset(X_train),
        dataset(X_test),
        models,
    )


def make_box(model, reference, candidates):
    box = Box.__new__(Box)
    box.model = model
    box.features = ("x0", "x1", "x2", "x3")
    box.target = ("y",)

    pipeline = SimpleNamespace(
        data_in=None,
        data_out=reference,
        column_transforms={},
    )

    box.pipeline = pipeline

    # Keep this test independent of Aspect transformation.
    box._pre_predict = lambda **kwargs: (
        candidates,
        pipeline,
    )

    return box


def direct_score(model, X_train, X_test):
    adapter = get_adapter(model)

    builder = LeafReferenceBuilder()
    builder.update(
        adapter.leaf_indices(model, X_train)
    )

    return builder.finish().inverse_score(
        adapter.leaf_indices(model, X_test)
    )


def test_single_model_occupancy(datasets_and_models):
    X_train, X_test, train, test, models = datasets_and_models

    model = models["rf"]
    box = make_box(model, train, test)

    result = box.predict(
        score="inverse_leaf_occupancy",
        batch_size=7,
        map_kwargs={"load_from_cache_file": False},
    )

    expected = direct_score(model, X_train, X_test)

    np.testing.assert_allclose(
        result["score:inverse_leaf_occupancy"],
        expected,
    )

    assert np.all(np.asarray(expected) > 0)
    assert np.all(np.asarray(expected) <= 1)

    np.testing.assert_allclose(
        result["prediction:y"],
        model.predict(X_test),
    )


def test_heterogeneous_ensemble(datasets_and_models):
    X_train, X_test, train, test, models = datasets_and_models

    ensemble = Ensemble(
        models,
        weights=(1.0, 3.0),
    )

    box = make_box(ensemble, train, test)

    result = box.predict(
        score="inverse_leaf_occupancy",
        disaggregate=True,
        batch_size=6,
        map_kwargs={"load_from_cache_file": False},
    )

    individual = np.column_stack([
        direct_score(model, X_train, X_test)
        for model in models.values()
    ])

    expected = np.average(
        individual,
        axis=1,
        weights=ensemble.weights,
    )

    np.testing.assert_allclose(
        result["score:inverse_leaf_occupancy"],
        expected,
    )

    for name, model in models.items():
        np.testing.assert_allclose(
            result[f"prediction:y:{name}"],
            model.predict(X_test),
        )


def test_batch_size_independence(datasets_and_models):
    _, _, train, test, models = datasets_and_models

    box = make_box(models["rf"], train, test)

    small = box.predict(
        score="inverse_leaf_occupancy",
        batch_size=3,
        map_kwargs={"load_from_cache_file": False},
    )

    large = box.predict(
        score="inverse_leaf_occupancy",
        batch_size=17,
        map_kwargs={"load_from_cache_file": False},
    )

    np.testing.assert_allclose(
        small["score:inverse_leaf_occupancy"],
        large["score:inverse_leaf_occupancy"],
    )


def test_reference_prepared_once(
    datasets_and_models,
    monkeypatch,
):
    _, _, train, test, models = datasets_and_models
    box = make_box(models["rf"], train, test)

    from arvore.models.adapters import SklearnAdapter

    original = SklearnAdapter.leaf_indices
    calls = []

    def counted(self, model, X):
        calls.append(len(X))
        return original(self, model, X)

    monkeypatch.setattr(
        SklearnAdapter,
        "leaf_indices",
        counted,
    )

    box.predict(
        score="inverse_leaf_occupancy",
        batch_size=4,
        map_kwargs={"load_from_cache_file": False},
    )

    # 48 reference rows in 12 batches;
    # 17 candidate rows in 5 batches.
    assert len(calls) == 17
    assert sum(calls) == 48 + 17


def test_missing_training_reference(
    datasets_and_models,
):
    _, _, _, test, models = datasets_and_models

    box = make_box(models["rf"], None, test)

    with pytest.raises(ValueError, match="training data"):
        box.predict(score="inverse_leaf_occupancy")
