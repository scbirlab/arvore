import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from aspect.data import DataPipeline

from arvore import Box

MODEL = {
    "class_path": "sklearn.ensemble.RandomForestRegressor",
    "init_args": {"n_estimators": 5, "random_state": 0},
}
TREE = ("sklearn.tree._tree.Tree",)


@pytest.fixture
def training(tmp_path):
    flags = [i % 2 == 0 for i in range(20)]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({
        "x": [float(i) for i in range(1, 21)],
        "has_ortholog": flags,
        "maybe": pa.array([None if i % 7 == 0 else f for i, f in enumerate(flags)]),
        "y": [2. * f + 0.1 * i for i, f in enumerate(flags)],
        "y_missing": pa.array([None if i % 5 == 0 else float(f) for i, f in enumerate(flags)]),
    }), path)
    return path


@pytest.mark.parametrize("config", [
    {"features": ["has_ortholog"], "pipeline": {}},
    {"features": ["has_ortholog", "log_x"], "pipeline": {"log_x": ["x", "log"]}},
    {"features": ["maybe"], "pipeline": {}},
    {"features": ["x"], "target": "has_ortholog", "pipeline": {}},
    {"features": ["x"], "target": "y_missing", "loss": {"masked": True}, "pipeline": {}},
])
def test_boolean_and_nullable_columns(tmp_path, training, config):
    box = Box.from_config({"model": MODEL, "target": "y"} | config)
    box.fit(str(training))
    predicted, scores = box.evaluate(str(training))
    assert np.all(np.isfinite(predicted[f"prediction:{box.target[0]}"]))

    box.save(tmp_path / "model.ao")
    restored = Box.load(tmp_path / "model.ao", trusted_types=TREE)
    np.testing.assert_allclose(
        restored.predict(str(training))[f"prediction:{box.target[0]}"],
        predicted[f"prediction:{box.target[0]}"],
    )


def test_load_does_not_featurize_training_data(tmp_path, training, monkeypatch):
    box = Box.from_config({
        "model": MODEL,
        "pipeline": {"log_x": ["x", "log"]},
        "features": ["log_x"],
        "target": "y",
    })
    box.fit(str(training))
    box.save(tmp_path / "model.ao")
    identity = box._training_identity

    calls = []
    original = DataPipeline.__call__

    def counting(self, data, *args, **kwargs):
        data_ = getattr(data, "data", data)
        calls.append(len(data_["x"]))
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(DataPipeline, "__call__", counting)
    restored = Box.load(tmp_path / "model.ao", trusted_types=TREE)
    assert calls == []
    assert restored._training_identity == identity
    assert restored.pipeline.data_in is not None

    restored.predict({"x": [1.5, 2.5]})
    assert calls == [2]

    # Featurized on first use, then reused.
    restored.evaluate()
    restored.predict()
    assert len(restored.training_data) == 20
    assert calls == [2, 20]
