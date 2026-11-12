"""Scorer caching and post-prediction warning tests."""

from copy import deepcopy
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor

from treeton import Box
from treeton.models.adapters import SklearnAdapter
from treeton.scorers import (
    EnsembleSD,
    InverseLeafOccupancy,
)
from treeton.scorers.cache import get_score_cache
from treeton.scorers.runner import ScoringRunner


@pytest.fixture
def fitted_box():
    rng = np.random.default_rng(42)

    x = rng.normal(size=48)
    y = x**2 + rng.normal(size=48) * 0.1

    box = Box(
        model=RandomForestRegressor(
            n_estimators=9,
            random_state=42,
        ),
        features=("x",),
        target=("y",),
        pipeline={},
    )

    box.fit(
        Dataset.from_dict({
            "x": x,
            "y": y,
        }),
        skip_eval=True,
    )

    return box


def test_leaf_reference_reused_across_preparations(
    fitted_box, monkeypatch
):
    box = fitted_box

    original = SklearnAdapter.leaf_indices
    calls = []

    def counted(self, estimator, X):
        calls.append(len(X))
        return original(self, estimator, X)

    monkeypatch.setattr(
        SklearnAdapter,
        "leaf_indices",
        counted,
    )

    raw = InverseLeafOccupancy()
    percentile = InverseLeafOccupancy(
        normalization="reference_percentile"
    )

    first = raw.prepare(box, batch_size=12)
    initial_calls = len(calls)

    assert initial_calls > 0

    second = raw.prepare(box, batch_size=5)
    assert second is first
    assert len(calls) == initial_calls

    calibrated = percentile.prepare(box, batch_size=12)
    assert len(calls) == initial_calls * 2
    assert calibrated[0].calibration_scores is not None

    again = percentile.prepare(box, batch_size=5)
    assert again is calibrated
    assert len(calls) == initial_calls * 2


def test_replacing_model_invalidates_cache(fitted_box):
    box = fitted_box
    scorer = InverseLeafOccupancy()

    old = scorer.prepare(box)
    box.model = deepcopy(box.model)

    new = scorer.prepare(box)

    assert new is not old


def test_refit_invalidates_cache(fitted_box):
    box = fitted_box
    scorer = InverseLeafOccupancy()

    scorer.prepare(box)
    assert get_score_cache(box)["leaf_reference"] is not None

    data = box.pipeline.data_in

    box.fit(data, skip_eval=True)

    assert box._scorer_cache is None

    scorer.prepare(box)
    assert get_score_cache(box)["leaf_reference"] is not None


def test_cached_reference_checkpoint_roundtrip(
    fitted_box, tmp_path
):
    box = fitted_box
    path = tmp_path / "model.tt"

    raw = InverseLeafOccupancy()
    percentile = InverseLeafOccupancy(
        normalization="reference_percentile"
    )

    original_raw = raw.prepare(box, batch_size=8)
    original_percentile = percentile.prepare(
        box, batch_size=8
    )

    box.save(path)

    assert (path / "leaf-reference.npz").is_file()

    restored = Box.load(
        path,
        trusted_types=("sklearn.tree._tree.Tree",),
    )

    cache = get_score_cache(restored)

    assert cache["leaf_reference"] is not None
    assert cache["leaf_percentile"] is not None

    restored_raw = raw.prepare(restored)
    restored_percentile = percentile.prepare(restored)

    assert restored_raw is cache["leaf_reference"]
    assert restored_percentile is cache["leaf_percentile"]

    for original, recovered in zip(
        original_raw, restored_raw
    ):
        assert original.n_reference == recovered.n_reference

        for a, b in zip(original.ids, recovered.ids):
            np.testing.assert_array_equal(a, b)

        for a, b in zip(original.counts, recovered.counts):
            np.testing.assert_array_equal(a, b)

    for original, recovered in zip(
        original_percentile, restored_percentile
    ):
        np.testing.assert_allclose(
            original.calibration_scores,
            recovered.calibration_scores,
        )


def test_cache_checksum_detects_corruption(
    fitted_box, tmp_path
):
    box = fitted_box
    path = tmp_path / "model.tt"

    InverseLeafOccupancy().prepare(box)
    box.save(path)

    cache_path = path / "leaf-reference.npz"
    with cache_path.open("ab") as stream:
        stream.write(b"corrupt")

    with pytest.raises(
        ValueError,
        match="checksum mismatch",
    ):
        Box.load(
            path,
            trusted_types=("sklearn.tree._tree.Tree",),
        )


def test_ensemble_sd_warns_only_if_everything_zero():
    box = SimpleNamespace(target=("y",))
    runner = ScoringRunner(
        box=box,
        scorers={"ensemble_sd": EnsembleSD()},
    )

    zero = Dataset.from_dict({
        "score:ensemble_sd:y": [0.0, 0.0, 0.0],
    })

    with pytest.warns(
        RuntimeWarning,
        match="disagreement is zero",
    ):
        runner.finalize(zero)

    nonzero = Dataset.from_dict({
        "score:ensemble_sd:y": [0.0, 0.01, 0.0],
    })

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        runner.finalize(nonzero)

    assert not recorded


def test_default_xgboost_repeat_zero_spread_warning():
    xgboost = pytest.importorskip("xgboost")

    from treeton import Ensemble

    rng = np.random.default_rng(123)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] ** 2 - X[:, 1]

    model = Ensemble.repeat(
        xgboost.XGBRegressor(
            n_estimators=12,
            max_depth=3,
            n_jobs=1,
        ),
        n=3,
        seed=42,
    )
    model.fit(X, y)

    box = Box.__new__(Box)
    box.model = model
    box.features = ("x0", "x1")
    box.target = ("y",)

    dataset = Dataset.from_dict({
        "x0": X[:10, 0],
        "x1": X[:10, 1],
    })

    pipeline = SimpleNamespace(column_transforms={})
    box._pre_predict = lambda **kwargs: (dataset, pipeline)

    with pytest.warns(
        RuntimeWarning,
        match="disagreement is zero",
    ):
        result = box.predict(
            score="ensemble_sd",
            batch_size=5,
            map_kwargs={"load_from_cache_file": False},
        )

    np.testing.assert_allclose(
        result["score:ensemble_sd:y"],
        0,
    )
