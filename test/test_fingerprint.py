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
