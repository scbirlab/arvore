"""Reference lifecycle, persistence, and concurrent preparation tests."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Lock
import json
import time
import warnings

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor

from treeton import Box
from treeton.scorers import EnsembleSD, InverseLeafOccupancy, TanimotoDistance
from treeton.scorers.distances import EuclideanReference
from treeton.scorers.references import ReferenceCache, reference_key
from treeton.scorers.runner import ScoringRunner


@pytest.fixture
def fitted_box():
    rng = np.random.default_rng(42)
    x = rng.normal(size=48)
    y = x ** 2 + rng.normal(size=48) * 0.1
    fingerprint = rng.integers(0, 2, (48, 64), dtype=np.uint8).tolist()
    data = Dataset.from_dict({
        "x": x, "y": y, "fingerprint": fingerprint,
    })
    box = Box(
        model=RandomForestRegressor(n_estimators=9, random_state=42),
        features=("x",), target=("y",), pipeline={},
    )
    box.fit(data, skip_eval=True)
    return box, data


def _predict(box, data, score):
    return box.predict(
        data, score=score, batch_size=12,
        map_kwargs={"load_from_cache_file": False},
    )


def _load(path):
    return Box.load(path, trusted_types=("sklearn.tree._tree.Tree",))


def test_reuses_reference_across_predictions_and_aliases(fitted_box, monkeypatch):
    box, data = fitted_box
    orig = EuclideanReference.build.__func__
    calls = []

    def counted(cls, X):
        calls.append(len(X))
        return orig(cls, X)

    monkeypatch.setattr(EuclideanReference, "build", classmethod(counted))
    _predict(box, data, "euclidean_distance")
    _predict(box, data, {
        "distance_1": "euclidean_distance",
        "distance_2": "euclidean_distance",
    })
    assert calls == [len(data)]
    assert len(box.references.snapshot()) == 1


def test_leaf_raw_and_percentile_share_leaf_counts(fitted_box):
    box, data = fitted_box
    _predict(box, data, "inverse_leaf_occupancy")
    assert len(box.references.snapshot()) == 1
    _predict(box, data, InverseLeafOccupancy("reference_percentile"))
    assert len(box.references.snapshot()) == 2
    _predict(box, data, "inverse_leaf_occupancy")
    assert len(box.references.snapshot()) == 2


def test_save_prepared_references_automatically(fitted_box, tmp_path):
    box, data = fitted_box
    original = _predict(box, data, [
        "euclidean_distance", "inverse_leaf_occupancy", "tanimoto_distance",
    ])
    count = len(box.references.snapshot())
    path = tmp_path / "model.tt"
    box.save(path)
    restored = _load(path)
    assert len(restored.references.snapshot()) == count

    # The restored Tanimoto/leaf references must be used directly. A
    # Euclidean NN estimator may be rebuilt from its persisted feature array.
    from treeton.scorers.tanimoto import TanimotoDistance as TD
    original_prepare = TD.prepare_reference
    try:
        TD.prepare_reference = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Tanimoto reference unexpectedly recomputed")
        )
        again = _predict(restored, data, [
            "euclidean_distance", "inverse_leaf_occupancy", "tanimoto_distance",
        ])
    finally:
        TD.prepare_reference = original_prepare
    for column in (
        "score:euclidean_distance",
        "score:inverse_leaf_occupancy",
        "score:tanimoto_distance",
    ):
        np.testing.assert_allclose(original[column], again[column])


def test_explicit_prepare_on_save(fitted_box, tmp_path):
    box, data = fitted_box
    path = tmp_path / "model.tt"
    box.save(path, score=[
        TanimotoDistance(column="fingerprint", query_chunk=1),
        InverseLeafOccupancy(normalization="reference_percentile"),
    ])
    assert len(box.references.snapshot()) == 3  # Tanimoto + counts + percentile
    restored = _load(path)
    assert len(restored.references.snapshot()) == 3
    scored = _predict(restored, data, [
        "tanimoto_distance",
        InverseLeafOccupancy(normalization="reference_percentile"),
    ])
    assert "score:tanimoto_distance" in scored.column_names


def test_distinct_reference_configs(fitted_box):
    box, data = fitted_box
    first = TanimotoDistance(column="fingerprint", query_chunk=1)
    second = TanimotoDistance(column="fingerprint", query_chunk=32)
    third = TanimotoDistance(column="fingerprint", structure_column="alternate")
    first.prepare(box)
    second.prepare(box)
    assert len(box.references.snapshot()) == 1  # Query chunk is not an index parameter
    third.prepare(box)
    assert len(box.references.snapshot()) == 2


def test_refit_clears_old_references(fitted_box):
    box, data = fitted_box
    _predict(box, data, "euclidean_distance")
    assert len(box.references.snapshot()) == 1
    box.fit(data, skip_eval=True)
    assert not box.references.snapshot()
    _predict(box, data, "euclidean_distance")
    assert len(box.references.snapshot()) == 1


def test_failed_build_is_not_cached():
    cache = ReferenceCache()
    spec = {"type": "test", "version": 1}
    def fail():
        raise RuntimeError("reference preparation failed")
    with pytest.raises(RuntimeError, match="failed"):
        cache.get_or_build(spec, fail)
    assert not cache.snapshot()
    assert cache.get_or_build(spec, lambda: 42) == 42


def test_concurrent_first_use_builds_once():
    cache = ReferenceCache()
    spec = {"type": "test", "version": 1}
    barrier = Barrier(8)
    lock = Lock()
    calls = []
    obj = object()

    def build():
        with lock:
            calls.append(1)
        time.sleep(0.01)
        return obj

    def prepare(_):
        barrier.wait()
        return cache.get_or_build(spec, build)

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(prepare, range(8)))
    assert len(calls) == 1
    assert all(value is obj for value in values)


def test_keys_are_canonical_and_not_alias_based():
    first = {"type": "euclidean", "version": 1, "metric": "euclidean"}
    other_order = {"metric": "euclidean", "version": 1, "type": "euclidean"}
    changed = {"type": "euclidean", "version": 1, "metric": "cityblock"}
    assert reference_key(first) == reference_key(other_order)
    assert reference_key(first) != reference_key(changed)


def test_corruption_and_incompatible_metadata(fitted_box, tmp_path):
    box, _ = fitted_box
    path = tmp_path / "model.tt"
    box.save(path, score="euclidean_distance")
    config_path = path / "config.json"
    manifest = json.loads(config_path.read_text())
    entry = manifest["artefacts"]["references"]["items"][0]
    resource = path / "references" / entry["filename"]
    resource.write_bytes(resource.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load(path)

    # Recreate valid archive, then invalidate the specification.
    box.save(path, score="euclidean_distance")
    manifest = json.loads(config_path.read_text())
    manifest["artefacts"]["references"]["items"][0]["spec"]["version"] = 99
    config_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible reference"):
        _load(path)


def test_modified_fitted_box_config_rejected(fitted_box, tmp_path):
    box, _ = fitted_box
    path = tmp_path / "model.tt"
    box.save(path, score="euclidean_distance")
    manifest_path = path / "config.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["features"] = ["changed"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fitted-state checksum"):
        _load(path)


def test_explicit_nonpersistable_reference_rejected(fitted_box, tmp_path):
    box, _ = fitted_box

    class UnsafeReferenceScorer:
        name = "unsafe_reference"

        def reference_specs(self, box):
            return ({"type": "opaque", "version": 1},)

        def prepare(self, box, *, batch_size):
            return box.references.get_or_build(self.reference_specs(box)[0], object)

        def __call__(self, context):
            return np.zeros(len(context.X))

    with pytest.raises(ValueError, match="safe persistence codec"):
        box.save(tmp_path / "no.tt", score=UnsafeReferenceScorer())
    assert not (tmp_path / "no.tt").exists()


def test_ensemble_sd_warns_only_if_everything_zero():
    from types import SimpleNamespace
    runner = ScoringRunner(
        box=SimpleNamespace(target=("y",)),
        scorers={"ensemble_sd": EnsembleSD()},
    )
    zero = Dataset.from_dict({"score:ensemble_sd:y": [0.0, 0.0, 0.0]})
    with pytest.warns(RuntimeWarning, match="disagreement is zero"):
        runner.finalize(zero)
    nonzero = Dataset.from_dict({"score:ensemble_sd:y": [0.0, 0.01, 0.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.finalize(nonzero)
    assert not caught


def test_default_xgboost_repeat_zero_spread_warning():
    xgboost = pytest.importorskip("xgboost")
    from types import SimpleNamespace
    from treeton import Ensemble

    rng = np.random.default_rng(123)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] ** 2 - X[:, 1]
    model = Ensemble.repeat(
        xgboost.XGBRegressor(n_estimators=12, max_depth=3, n_jobs=1),
        n=3, seed=42,
    )
    model.fit(X, y)
    box = Box.__new__(Box)
    box.model = model
    box.features = ("x0", "x1")
    box.target = ("y",)
    dataset = Dataset.from_dict({"x0": X[:10, 0], "x1": X[:10, 1]})
    pipeline = SimpleNamespace(column_transforms={})
    box._pre_predict = lambda **kwargs: (dataset, pipeline)
    with pytest.warns(RuntimeWarning, match="disagreement is zero"):
        result = box.predict(
            score="ensemble_sd", batch_size=5,
            map_kwargs={"load_from_cache_file": False},
        )
    np.testing.assert_allclose(result["score:ensemble_sd:y"], 0)



def test_tanimoto_numpy2_matches_brute_force():
    from treeton.scorers.tanimoto import TanimotoReference

    rng = np.random.default_rng(123)
    references = rng.integers(0, 2, (37, 130), dtype=np.uint8)
    queries = rng.integers(0, 2, (13, 130), dtype=np.uint8)

    # Include empty fingerprints and identical references.
    references[0] = 0
    queries[0] = 0
    queries[1] = references[1]

    ref = TanimotoReference.build(references)

    expected = []
    for query in queries:
        intersection = np.logical_and(
            references, query
        ).sum(axis=1)
        union = np.logical_or(
            references, query
        ).sum(axis=1)

        similarity = np.divide(
            intersection,
            union,
            out=np.ones(len(references), dtype=float),
            where=union != 0,
        )
        expected.append(1 - similarity)

    expected = np.asarray(expected)

    for k in (1, 2, 4, 8):
        observed = ref.distance(
            queries,
            n_neighbors=k,
            query_chunk=3,
            reference_chunk=5,
        )
        truth = np.sort(expected, axis=1)[:, :k].mean(axis=1)
        np.testing.assert_allclose(
            observed, truth, rtol=0, atol=1e-12
        )


def test_euclidean_k_neighbours():
    from treeton.scorers.distances import EuclideanReference

    reference = EuclideanReference.build(
        np.array([[0.0], [2.0], [5.0]])
    )
    queries = np.array([[1.0], [4.0]])

    np.testing.assert_allclose(
        reference.distance(queries, n_neighbors=1),
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        reference.distance(queries, n_neighbors=2),
        [1.0, 1.5],
    )


def test_neighbour_count_does_not_duplicate_references(fitted_box):
    from treeton.scorers.distances import EuclideanDistance

    box, data = fitted_box

    scorers = [
        EuclideanDistance(n_neighbors=1),
        EuclideanDistance(n_neighbors=4),
        TanimotoDistance(n_neighbors=1),
        TanimotoDistance(n_neighbors=4),
    ]

    for scorer in scorers:
        scorer.prepare(box)

    assert len(box.references.snapshot()) == 2

    result = _predict(
        box, data,
        {
            "euclidean_1": scorers[0],
            "euclidean_4": scorers[1],
            "tanimoto_1": scorers[2],
            "tanimoto_4": scorers[3],
        },
    )

    assert len(result) == len(data)
    assert len(box.references.snapshot()) == 2

    # Mean k-neighbour distance cannot be smaller
    # than the nearest-neighbour distance.
    assert np.all(
        np.asarray(result["score:euclidean_4"])
        >= np.asarray(result["score:euclidean_1"])
    )
    assert np.all(
        np.asarray(result["score:tanimoto_4"])
        >= np.asarray(result["score:tanimoto_1"])
    )


def test_reference_cache_retains_specification():
    from treeton.scorers.references import (
        ReferenceCache, reference_key,
    )

    cache = ReferenceCache()
    spec = {"type": "test", "version": 1}
    value = object()

    assert cache.get_or_build(spec, lambda: value) is value

    key = reference_key(spec)
    assert cache.snapshot()[key] == (spec, value)

    restored = ReferenceCache()
    restored.restore(cache.snapshot())

    assert restored.get_or_build(
        spec,
        lambda: pytest.fail("Unexpected rebuild"),
    ) is value


def test_save_tracks_actual_reference_requests(fitted_box, tmp_path):
    """A scorer need not separately declare its reference keys."""
    from treeton.scorers.distances import EuclideanDistance

    box, data = fitted_box

    class CustomDistance:
        name = "custom_distance"

        def prepare(self, box, *, batch_size):
            return EuclideanDistance().prepare(
                box, batch_size=batch_size
            )

        def __call__(self, context):
            return context.state.distance(context.X)

    path = tmp_path / "tracked.tt"

    # First request prepares the resource.
    box.save(path, score=CustomDistance())
    assert len(box.references.snapshot()) == 1

    # Repeated preparation must also track cache hits.
    box.save(path, score=CustomDistance())

    restored = _load(path)
    assert len(restored.references.snapshot()) == 1

    result = _predict(restored, data, "euclidean_distance")
    assert "score:euclidean_distance" in result.column_names
