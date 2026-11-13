"""Regression tests for configurable reference distances and neighbour labels."""

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances

from treeton import Box
from treeton.scorers import FeatureDistance, TanimotoDistance
from treeton.scorers.distances import FeatureReference
from treeton.scorers.tanimoto import TanimotoReference


@pytest.fixture
def fitted_box():
    rng = np.random.default_rng(42)
    n = 24
    smiles = [f"molecule_{i}" for i in range(n)]
    fingerprints = rng.integers(
        0, 2, (n, 16), dtype=np.uint8
    ).tolist()

    data = Dataset.from_dict({
        "x": np.linspace(0, 10, n),
        "y": rng.normal(size=n),
        "fingerprint": fingerprints,
        "smiles": smiles,
    })

    box = Box(
        model=RandomForestRegressor(
            n_estimators=8, random_state=42
        ),
        features=("x",),
        target=("y",),
        pipeline={},
    )
    box.fit(data, skip_eval=True)
    return box, data


def predict(box, data, score):
    return box.predict(
        data,
        score=score,
        batch_size=5,
        map_kwargs={"load_from_cache_file": False},
    )


def load(path):
    return Box.load(
        path,
        trusted_types=("sklearn.tree._tree.Tree",),
    )


@pytest.mark.parametrize("metric", [
    "euclidean", "manhattan", "chebyshev", "cosine"
])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_feature_distance_matches_sklearn(metric, k):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(32, 5))
    Q = rng.normal(size=(7, 5))

    reference = FeatureReference.build(X, metric=metric)
    actual = reference.distance(Q, n_neighbors=k)

    expected = np.sort(
        pairwise_distances(Q, X, metric=metric),
        axis=1,
    )[:, :k].mean(axis=1)

    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-10
    )


def test_feature_distance_reuses_index_across_k(fitted_box):
    box, data = fitted_box

    result = predict(box, data, {
        "near": FeatureDistance(n_neighbors=1),
        "four": FeatureDistance(n_neighbors=4),
    })

    assert len(box.references.snapshot()) == 1
    assert np.all(
        np.asarray(result["score:four"])
        >= np.asarray(result["score:near"]) - 1e-12
    )


def test_feature_metric_changes_reference_key(fitted_box):
    box, data = fitted_box

    result = predict(box, data, {
        "euclidean": FeatureDistance(metric="euclidean"),
        "manhattan": FeatureDistance(metric="manhattan"),
    })

    assert len(box.references.snapshot()) == 2
    assert "score:euclidean" in result.column_names
    assert "score:manhattan" in result.column_names


def test_tanimoto_neighbors_match_brute_force():
    refs = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    queries = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    reference = TanimotoReference.build(refs)

    for k in (1, 2, 3, 5):
        actual, indices = reference.distance(
            queries,
            n_neighbors=k,
            return_indices=True,
            query_chunk=2,
            reference_chunk=2,
        )

        expected_distances = []
        expected_indices = []

        for query in queries:
            intersection = np.logical_and(
                refs, query
            ).sum(axis=1)
            union = np.logical_or(
                refs, query
            ).sum(axis=1)

            similarity = np.divide(
                intersection,
                union,
                out=np.ones(len(refs), dtype=float),
                where=union != 0,
            )

            order = np.lexsort(
                (np.arange(len(refs)), -similarity)
            )[:k]

            expected_indices.append(order)
            expected_distances.append(
                1 - similarity[order].mean()
            )

        np.testing.assert_array_equal(
            indices, np.asarray(expected_indices)
        )
        np.testing.assert_allclose(
            actual, expected_distances,
            rtol=0, atol=1e-12,
        )


def test_tanimoto_scalar_return_is_unchanged():
    rng = np.random.default_rng(123)
    refs = rng.integers(0, 2, (30, 130), dtype=np.uint8)
    queries = rng.integers(0, 2, (8, 130), dtype=np.uint8)

    reference = TanimotoReference.build(refs)
    direct = reference.distance(queries, n_neighbors=4)
    with_indices, indices = reference.distance(
        queries, n_neighbors=4, return_indices=True
    )

    np.testing.assert_allclose(direct, with_indices)
    assert direct.shape == (8,)
    assert indices.shape == (8, 4)


def test_tanimoto_neighbor_output(fitted_box):
    box, data = fitted_box

    result = predict(box, data, {
        "tanimoto": TanimotoDistance(
            n_neighbors=4,
            return_neighbors=True,
        ),
    })

    assert "score:tanimoto" in result.column_names
    assert "score:tanimoto:neighbor" in result.column_names

    neighbors = result["score:tanimoto:neighbor"]
    assert len(neighbors) == len(data)
    assert all(len(value.split(";")) == 4 for value in neighbors)

    # Fingerprints and labels are separate resources.
    assert len(box.references.snapshot()) == 2


def test_tanimoto_k1_returns_single_label(fitted_box):
    box, data = fitted_box

    result = predict(box, data, {
        "tanimoto": TanimotoDistance(
            return_neighbors=True,
            n_neighbors=1,
        ),
    })

    labels = result["score:tanimoto:neighbor"]
    assert all(";" not in label for label in labels)
    assert all(label in data["smiles"] for label in labels)


def test_tanimoto_labels_are_prepared_lazily(fitted_box):
    box, data = fitted_box

    predict(box, data, "tanimoto_distance")
    assert len(box.references.snapshot()) == 1

    predict(box, data, TanimotoDistance(
        n_neighbors=4,
        return_neighbors=True,
    ))
    assert len(box.references.snapshot()) == 2


def test_tanimoto_missing_neighbor_column(fitted_box):
    box, data = fitted_box
    with pytest.raises(ValueError, match="neighbour column"):
        predict(box, data, TanimotoDistance(
            return_neighbors=True,
            neighbor_column="missing_identifier",
        ))


def test_tanimoto_radius_affects_resource_key(fitted_box):
    box, _ = fitted_box

    TanimotoDistance(radius=2).prepare(box)
    TanimotoDistance(radius=3).prepare(box)
    assert len(box.references.snapshot()) == 2


def test_checkpoint_preserves_features_and_labels(
    fitted_box, tmp_path, monkeypatch
):
    box, data = fitted_box
    scorers = {
        "feature": FeatureDistance(
            metric="manhattan", n_neighbors=3
        ),
        "tanimoto": TanimotoDistance(
            n_neighbors=3, return_neighbors=True
        ),
    }

    before = predict(box, data, scorers)

    path = tmp_path / "distance_model.tt"
    box.save(path)
    restored = load(path)

    assert len(restored.references.snapshot()) == 3

    # Rebuilding either expensive reference must be unnecessary.
    def unexpected(*args, **kwargs):
        raise AssertionError("Reference unexpectedly rebuilt")

    monkeypatch.setattr(
        FeatureReference, "build", unexpected
    )
    monkeypatch.setattr(
        TanimotoReference, "build", unexpected
    )

    after = predict(restored, data, scorers)

    for name in ("score:feature", "score:tanimoto"):
        np.testing.assert_allclose(
            before[name], after[name]
        )

    assert (
        before["score:tanimoto:neighbor"]
        == after["score:tanimoto:neighbor"]
    )


def test_explicit_save_prepares_neighbor_labels(
    fitted_box, tmp_path
):
    box, data = fitted_box
    path = tmp_path / "explicit.tt"

    box.save(
        path,
        score=TanimotoDistance(
            return_neighbors=True,
            n_neighbors=4,
        ),
    )

    restored = load(path)
    assert len(restored.references.snapshot()) == 2

    result = predict(restored, data, TanimotoDistance(
        return_neighbors=True,
        n_neighbors=4,
    ))
    assert "score:tanimoto_distance:neighbor" in result.column_names


def test_different_neighbor_label_columns_share_fingerprints(
    fitted_box
):
    box, data = fitted_box

    predict(box, data, {
        "one": TanimotoDistance(
            return_neighbors=True,
            neighbor_column="smiles",
        ),
        "two": TanimotoDistance(
            return_neighbors=True,
            neighbor_column="smiles",
            neighbor_delimiter="|",
        ),
    })

    assert len(box.references.snapshot()) == 2


def test_invalid_k_rejected():
    ref = TanimotoReference.build(
        np.array([[1, 0], [0, 1]], dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="neighbours"):
        ref.distance([[1, 0]], n_neighbors=3)

    feature = FeatureReference.build([[0], [1]])
    with pytest.raises(ValueError, match="neighbours"):
        feature.distance([[0]], n_neighbors=3)


@pytest.mark.parametrize("metric", [
    "euclidean", "manhattan", "chebyshev", "cosine"
])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_feature_variance_matches_brute_force(metric, k):
    rng = np.random.default_rng(450)
    X = rng.normal(size=(32, 5))
    Q = rng.normal(size=(9, 5))

    ref = FeatureReference.build(X, metric=metric)
    actual_mean, actual_var = ref.distance(
        Q, n_neighbors=k, return_var=True
    )

    expected = np.sort(
        pairwise_distances(Q, X, metric=metric),
        axis=1,
    )[:, :k]

    np.testing.assert_allclose(
        actual_mean, expected.mean(axis=1),
        rtol=1e-10, atol=1e-10,
    )
    np.testing.assert_allclose(
        actual_var, expected.var(axis=1, ddof=0),
        rtol=1e-10, atol=1e-10,
    )

    np.testing.assert_allclose(
        actual_mean, ref.distance(Q, n_neighbors=k)
    )

    if k == 1:
        np.testing.assert_array_equal(
            actual_var, np.zeros(len(Q))
        )


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("return_indices", [False, True])
def test_tanimoto_variance_matches_brute_force(k, return_indices):
    rng = np.random.default_rng(451)
    X = rng.integers(0, 2, (27, 130), dtype=np.uint8)
    Q = rng.integers(0, 2, (11, 130), dtype=np.uint8)

    X[0] = 0
    Q[0] = 0
    X[1] = X[2]
    Q[1] = X[1]

    ref = TanimotoReference.build(X)

    result = ref.distance(
        Q,
        n_neighbors=k,
        return_var=True,
        return_indices=return_indices,
        query_chunk=3,
        reference_chunk=5,
    )

    if return_indices:
        mean, variance, indices = result
        assert indices.shape == (len(Q), k)
    else:
        mean, variance = result

    expected_mean = []
    expected_var = []
    expected_idx = []

    for q in Q:
        intersection = np.logical_and(X, q).sum(axis=1)
        union = np.logical_or(X, q).sum(axis=1)
        similarities = np.divide(
            intersection,
            union,
            out=np.ones(len(X), dtype=float),
            where=union != 0,
        )
        order = np.lexsort(
            (np.arange(len(X)), -similarities)
        )[:k]
        selected = 1.0 - similarities[order]
        expected_mean.append(selected.mean())
        expected_var.append(selected.var(ddof=0))
        expected_idx.append(order)

    np.testing.assert_allclose(
        mean, expected_mean, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        variance, expected_var, rtol=0, atol=1e-12
    )
    if return_indices:
        np.testing.assert_array_equal(
            indices, expected_idx
        )

    np.testing.assert_allclose(
        mean, ref.distance(Q, n_neighbors=k),
        rtol=0, atol=1e-12,
    )

    if k == 1:
        np.testing.assert_array_equal(
            variance, np.zeros(len(Q))
        )


def test_variance_scorer_outputs_and_reference_reuse(fitted_box):
    box, data = fitted_box

    result = predict(box, data, {
        "feature_mean": FeatureDistance(
            n_neighbors=4,
        ),
        "feature_stats": FeatureDistance(
            n_neighbors=4,
            return_var=True,
        ),
        "tanimoto_mean": TanimotoDistance(
            n_neighbors=4,
        ),
        "tanimoto_stats": TanimotoDistance(
            n_neighbors=4,
            return_var=True,
            return_neighbors=True,
        ),
    })

    for name in ("feature_stats", "tanimoto_stats"):
        assert f"score:{name}" in result.column_names
        assert f"score:{name}:var" in result.column_names
        assert all(
            value >= -1e-15
            for value in result[f"score:{name}:var"]
        )

    np.testing.assert_allclose(
        result["score:feature_stats"],
        result["score:feature_mean"],
    )
    np.testing.assert_allclose(
        result["score:tanimoto_stats"],
        result["score:tanimoto_mean"],
    )

    assert "score:tanimoto_stats:neighbor" in result.column_names

    # One feature reference, one fingerprint reference,
    # and one label resource; variance adds no reference.
    assert len(box.references.snapshot()) == 3


def test_variance_checkpoint_roundtrip(fitted_box, tmp_path):
    box, data = fitted_box
    scorers = {
        "feature": FeatureDistance(
            metric="manhattan",
            n_neighbors=3,
            return_var=True,
        ),
        "tanimoto": TanimotoDistance(
            n_neighbors=3,
            return_var=True,
            return_neighbors=True,
        ),
    }

    before = predict(box, data, scorers)

    path = tmp_path / "variance.tt"
    box.save(path)
    restored = load(path)
    after = predict(restored, data, scorers)

    for column in (
        "score:feature",
        "score:feature:var",
        "score:tanimoto",
        "score:tanimoto:var",
    ):
        np.testing.assert_allclose(
            before[column], after[column],
            rtol=0, atol=1e-12,
        )

    assert (
        before["score:tanimoto:neighbor"]
        == after["score:tanimoto:neighbor"]
    )
    assert len(restored.references.snapshot()) == 3


def test_return_var_single_neighbor_zero(fitted_box):
    box, data = fitted_box

    result = predict(box, data, {
        "feature": FeatureDistance(
            n_neighbors=1,
            return_var=True,
        ),
        "tanimoto": TanimotoDistance(
            n_neighbors=1,
            return_var=True,
        ),
    })

    np.testing.assert_array_equal(
        result["score:feature:var"],
        np.zeros(len(data)),
    )
    np.testing.assert_array_equal(
        result["score:tanimoto:var"],
        np.zeros(len(data)),
    )
