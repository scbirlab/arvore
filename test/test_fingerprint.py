import warnings

import numpy as np
import pytest
from datasets import Dataset

from arvore import Box

TRUSTED = ("sklearn.tree._tree.Tree",)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return Dataset.from_dict({
        "a": X[:, 0], "b": X[:, 1], "c": X[:, 2],
        "y": X.sum(axis=1) + rng.normal(size=n),
    })


def _box(class_path="sklearn.ensemble.RandomForestRegressor", **init):
    init = {"n_estimators": 10, "random_state": 0, **init}
    return Box.from_config({
        "model": {"class_path": class_path, "init_args": init},
        "pipeline": {},
        "features": ["a", "b", "c"],
        "target": "y",
    })


@pytest.mark.parametrize("class_path", [
    "sklearn.ensemble.RandomForestRegressor",
    "xgboost.XGBRegressor",
    "lightgbm.LGBMRegressor",
])
def test_fingerprint_is_content_identity(tmp_path, class_path):
    pytest.importorskip(class_path.split(".")[0])
    init = {"verbose": -1} if "lightgbm" in class_path else {}
    train = _data()

    box = _box(class_path, **init).fit(train, skip_eval=True)
    twin = _box(class_path, **init).fit(train, skip_eval=True)
    other = _box(class_path, **init).fit(_data(seed=1), skip_eval=True)

    assert box.fingerprint == twin.fingerprint
    assert box.fingerprint != other.fingerprint

    box.save(tmp_path / "m.ao")
    loaded = Box.load(tmp_path / "m.ao", trusted_types=TRUSTED)
    assert loaded.fingerprint == box.fingerprint


def test_refit_resets_fingerprint():
    box = _box().fit(_data(), skip_eval=True)
    before = box.fingerprint
    box.fit(_data(seed=1), skip_eval=True)
    assert box.fingerprint != before


def test_predict_fingerprint_tracks_inputs_and_scorers():
    box = _box().fit(_data(), skip_eval=True)
    candidates = _data(50, seed=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no datasets hashing fallback
        first = box.predict(candidates, score=["feature_distance"])
        again = box.predict(candidates, score=["feature_distance"])
        other = box.predict(candidates, score=["tree_sd"])
        plain = box.predict(candidates)

    assert first._fingerprint == again._fingerprint
    assert len({
        first._fingerprint, other._fingerprint, plain._fingerprint,
    }) == 3


def test_skops_digest_ignores_ids_and_padding():
    import skops.io as sio
    from sklearn.ensemble import RandomForestRegressor
    from arvore.models.adapters import _skops_digest

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    model = RandomForestRegressor(5, random_state=0).fit(X, X.sum(axis=1))
    raw = sio.dumps(model)
    restored = sio.loads(raw, trusted=sio.get_untrusted_types(data=raw))

    assert _skops_digest(raw) == _skops_digest(sio.dumps(restored))


def test_identity_never_materialises_training_data(tmp_path, monkeypatch):
    from arvore.box.identity import training_identity

    def refuse(*args, **kwargs):
        raise AssertionError("training data materialised")

    monkeypatch.setattr(Dataset, "to_pandas", refuse)

    box = _box().fit(_data(), skip_eval=True)
    assert training_identity(box.pipeline) == box._training_identity
    assert box.fingerprint

    box.save(tmp_path / "m.ao")
    assert Box.load(tmp_path / "m.ao", trusted_types=TRUSTED).fingerprint == box.fingerprint


def test_training_identity_reads_only_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from providence import DataResolver
    from arvore.box.identity import training_identity

    _data().to_csv(tmp_path / "train.csv")
    from_file = DataResolver(str(tmp_path / "train.csv"))
    from_mapping = DataResolver({"x": [1., 2., 3.]})
    in_memory = DataResolver(_data())

    def refuse(*args, **kwargs):
        raise AssertionError("training data materialised")

    monkeypatch.setattr(Dataset, "to_pandas", refuse)

    for resolver in (from_file, from_mapping):
        assert training_identity(SimpleNamespace(data_in=resolver)) == {
            "checksum": resolver.data_source.checksum,
            "selected_rows": None,
        }
    assert training_identity(SimpleNamespace(data_in=in_memory)) == {
        "fingerprint": in_memory.data._fingerprint,
        "selected_rows": None,
    }


def test_checkpointed_transformed_data_is_not_recomputed(tmp_path, monkeypatch):
    from aspect import DataPipeline

    box = Box.from_config({
        "model": {
            "class_path": "sklearn.ensemble.RandomForestRegressor",
            "init_args": {"n_estimators": 5, "random_state": 0},
        },
        "pipeline": {"column_transforms": {"a2": ["a", "log"]}},
        "features": ["a2", "b", "c"],
        "target": "y",
    })
    training = _data()
    box.fit(training.map(lambda row: {"a": abs(row["a"]) + 1.}), skip_eval=True)
    box.save(tmp_path / "m.ao", save_transformed=True)

    def refuse(*args, **kwargs):
        raise AssertionError("pipeline re-run")

    monkeypatch.setattr(DataPipeline, "__call__", refuse)
    restored = Box.load(tmp_path / "m.ao", trusted_types=TRUSTED)

    assert restored.fingerprint == box.fingerprint
    assert len(restored.training_data) == len(training)
    np.testing.assert_allclose(
        restored.predict()["prediction:y"],
        box.predict()["prediction:y"],
    )


def _resolver(source, **data):
    from types import SimpleNamespace
    return SimpleNamespace(
        data_in=SimpleNamespace(
            data_source=source,
            data=SimpleNamespace(**data),
            selected_rows=None,
        )
    )


@pytest.mark.parametrize(
    "uri,revision,expected",
    [
        ("hf://datasets/org/data@" + "a" * 40, "a" * 40, {"uri", "revision"}),
        ("hf://datasets/org/data@main", "main", {"fingerprint"}),
        ("hf://datasets/org/data", None, {"fingerprint"}),
    ],
)
def test_only_pinned_remote_sources_identify_data(uri, revision, expected):
    from providence import DataSource
    from arvore.box.identity import training_identity

    identity = training_identity(
        _resolver(DataSource(uri=uri, revision=revision), _fingerprint="abc")
    )

    assert set(identity) - {"selected_rows"} == expected


def test_unidentifiable_data_never_shares_identity():
    from providence import DataSource
    from arvore.box.identity import training_identity

    pipeline = _resolver(DataSource())

    first, second = training_identity(pipeline), training_identity(pipeline)
    assert set(first) == {"unidentified", "selected_rows"}
    assert first != second


def test_training_identity_uses_providence_provenance(tmp_path):
    from providence import DataResolver

    _data().to_csv(tmp_path / "train.csv")
    (tmp_path / "copy.csv").write_bytes((tmp_path / "train.csv").read_bytes())

    from_file = _box().fit(str(tmp_path / "train.csv"), skip_eval=True)
    from_copy = _box().fit(str(tmp_path / "copy.csv"), skip_eval=True)
    subset = _box().fit(
        DataResolver(str(tmp_path / "train.csv")).select_rows(0, 150),
        skip_eval=True,
    )
    ephemeral = _box().fit(_data(), skip_eval=True)

    assert set(from_file._training_identity) == {"checksum", "selected_rows"}
    assert from_file._training_identity == from_copy._training_identity
    assert from_file.fingerprint == from_copy.fingerprint
    assert subset._training_identity["selected_rows"] == [[0, 150]]
    assert subset.fingerprint != from_file.fingerprint
    assert set(ephemeral._training_identity) == {"fingerprint", "selected_rows"}

    from_file.save(tmp_path / "file.ao")
    assert Box.load(tmp_path / "file.ao", trusted_types=TRUSTED).fingerprint == from_file.fingerprint


@pytest.mark.parametrize("file_backed", [False, True])
def test_tampered_training_data_fails_to_load(tmp_path, file_backed):
    training = _data()
    if file_backed:
        training.to_csv(tmp_path / "train.csv")
        training = str(tmp_path / "train.csv")
    box = _box().fit(training, skip_eval=True)
    box.save(tmp_path / "m.ao")
    (embedded,) = (tmp_path / "m.ao").rglob("data.parquet")
    _data(seed=1).to_parquet(str(embedded))

    with pytest.raises(IOError, match="Checksum"):
        Box.load(tmp_path / "m.ao", trusted_types=TRUSTED)
