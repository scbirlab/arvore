"""Censored and masked losses across model adapters and Box checkpoints."""

import numpy as np
import pytest
from datasets import Dataset
from sklearn.ensemble import RandomForestRegressor

from arvore import Box, Loss
from arvore.box.base import _exact_observed
from arvore.models import get_adapter
from arvore.models.adapters import load_model, save_model
from arvore.models.losses import LOSSES, resolve_loss, squared_error


def absolute_interval_error(prediction, lower, upper):
    """Absolute distance to the target interval.

    Its Hessian of one is a surrogate: the true Hessian is zero almost
    everywhere, which gives boosting no curvature to step with.
    """
    residual = prediction - np.clip(prediction, lower, upper)
    return np.abs(residual), np.sign(residual), np.ones_like(residual)


CUSTOM_LOSS = f"{__name__}.absolute_interval_error"


def make_censored(seed, n=600, fraction=0.4, max_offset=4.):
    """Right-censored rows report a value below the truth."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    truth = 10. * X[:, 0] + rng.normal(0., 0.3, n)
    censor = (rng.random(n) < fraction).astype(int)
    y = np.where(
        censor == 1,
        truth - rng.uniform(0., max_offset, n),
        truth,
    )
    return X, truth, y, censor


@pytest.fixture
def censored():
    return make_censored(0), make_censored(1, n=200)


def boosters(n_estimators=300):
    xgb = pytest.importorskip("xgboost")
    lgb = pytest.importorskip("lightgbm")
    return [
        xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=3, learning_rate=0.1,
            n_jobs=1, random_state=42,
        ),
        lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.1, verbosity=-1,
            n_jobs=1, random_state=42,
        ),
    ]


def test_squared_error_on_intervals():
    inf = np.inf
    prediction = np.full(6, 1.)
    lower = np.array([3., 0., 3., -inf, -inf, -inf])
    upper = np.array([3., inf, inf, 0., 3., inf])

    loss, gradient, hessian = squared_error(prediction, lower, upper)

    np.testing.assert_allclose(loss, [2., 0., 2., .5, 0., 0.])
    np.testing.assert_allclose(gradient, [-2., 0., -2., 1., 0., 0.])
    np.testing.assert_allclose(hessian, [1., 0., 1., 1., 0., 0.])


def test_bounds_from_censoring_and_mask():
    y = np.array([1., 2., 3., np.nan])
    lower, upper = Loss(masked=True).bounds(y, censor=[0, -1, 2, 0])

    np.testing.assert_array_equal(lower, [1., -np.inf, 3., -np.inf])
    np.testing.assert_array_equal(upper, [1., 2., np.inf, np.inf])


def test_mean_ignores_masked_targets():
    loss = Loss(masked=True)
    lower, upper = loss.bounds(np.array([[0., np.nan], [2., 2.]]))
    prediction = np.array([[1., 100.], [2., 4.]])

    assert loss.mean(prediction, lower, upper) == pytest.approx(2.5 / 3)
    assert loss.mean(
        prediction, lower, upper, weight=[3., 1.],
    ) == pytest.approx(3.5 / 5)


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, None),
        ("squared_error", Loss()),
        ({"masked": True}, Loss(masked=True)),
        ({"name": CUSTOM_LOSS, "masked": True}, Loss(CUSTOM_LOSS, masked=True)),
        (absolute_interval_error, Loss(name=CUSTOM_LOSS)),
    ],
)
def test_resolve_loss(spec, expected):
    loss = resolve_loss(spec)

    assert loss == expected
    if loss is not None:
        assert resolve_loss(loss.to_dict()) == loss


def test_function_resolution():
    assert Loss().function is LOSSES["squared_error"]
    assert resolve_loss(absolute_interval_error).function is absolute_interval_error

    with pytest.raises((AttributeError, ImportError)):
        resolve_loss(lambda prediction, lower, upper: None)

    with pytest.raises(TypeError, match="loss must be"):
        resolve_loss(1.)


@pytest.mark.parametrize(
    "loss,censor,masked_y,expected",
    [
        (Loss(), None, False, False),
        (Loss(), [0, 1], False, True),
        (Loss(masked=True), None, True, True),
        (Loss(), None, True, False),
        (Loss(CUSTOM_LOSS), None, False, True),
    ],
)
def test_requires_objective(loss, censor, masked_y, expected):
    y = np.array([1., np.nan if masked_y else 2.])

    assert loss.requires_objective(y, censor) is expected


def test_censored_loss_removes_bias(censored):
    (X, _, y, censor), (X_val, truth_val, y_val, censor_val) = censored

    for model in boosters():
        adapter = get_adapter(model)
        naive = type(model)(**model.get_params())

        adapter.fit(
            model, X, y,
            censor=censor,
            validation=(X_val, y_val, censor_val),
            early_stopping_rounds=20,
        )
        adapter.fit(naive, X, y)

        censored_bias = np.mean(model.predict(X_val) - truth_val)
        naive_bias = np.mean(naive.predict(X_val) - truth_val)

        assert abs(censored_bias) < 0.2
        assert naive_bias < -0.5


def test_custom_objective_restores_estimator_parameters(censored):
    (X, _, y, censor), _ = censored

    for model in boosters():
        before = model.get_params()
        get_adapter(model).fit(model, X, y, censor=censor)

        assert model.get_params().items() >= before.items()


def test_custom_objective_checkpoint(tmp_path, censored):
    (X, _, y, censor), (X_val, _, y_val, censor_val) = censored

    for model in boosters(n_estimators=1000):
        get_adapter(model).fit(
            model, X, y,
            censor=censor,
            validation=(X_val, y_val, censor_val),
            early_stopping_rounds=5,
        )
        before = model.predict(X_val)

        best_iteration = getattr(model, "best_iteration_", None) or model.best_iteration
        assert 0 < best_iteration < 999

        directory = tmp_path / type(model).__name__
        restored = load_model(directory, save_model(model, directory))

        np.testing.assert_allclose(restored.predict(X_val), before, rtol=1e-6)


def test_custom_objective_refit(censored):
    (X, _, y, censor), _ = censored

    for model, fresh_censored, fresh_plain in zip(
        boosters(50), boosters(50), boosters(50),
    ):
        adapter = get_adapter(model)
        adapter.fit(model, X, y, censor=censor)
        adapter.fit(fresh_censored, X, y, censor=censor)
        np.testing.assert_allclose(model.predict(X), fresh_censored.predict(X))

        adapter.fit(model, X, y)
        adapter.fit(fresh_plain, X, y)
        np.testing.assert_allclose(model.predict(X), fresh_plain.predict(X))

        adapter.fit(model, X, y, censor=censor)
        np.testing.assert_allclose(model.predict(X), fresh_censored.predict(X))


def test_censored_early_stopping_uses_loss(censored):
    (X, _, y, censor), (X_val, _, y_val, censor_val) = censored

    for model in boosters():
        get_adapter(model).fit(
            model, X, y,
            censor=censor,
            validation=(X_val, y_val, censor_val),
            early_stopping_rounds=5,
        )

        (history,) = model.evals_result_.values()
        assert "squared_error" in history


def test_custom_loss_always_used(censored):
    (X, _, y, _), _ = censored

    for model in boosters(50):
        adapter = get_adapter(model)
        native = type(model)(**model.get_params())

        adapter.fit(model, X, y, loss=Loss(CUSTOM_LOSS))
        adapter.fit(native, X, y, loss=Loss())

        assert not np.allclose(model.predict(X), native.predict(X))


def test_masked_single_target_drops_missing_rows():
    X, _, y, _ = make_censored(0, n=100)
    y = y.copy()
    y[::5] = np.nan
    weight = np.linspace(1., 2., len(y))
    observed = np.isfinite(y)

    masked = RandomForestRegressor(n_estimators=5, random_state=42)
    get_adapter(masked).fit(
        masked, X, y,
        loss=Loss(masked=True),
        fit_kwargs={"sample_weight": weight},
    )
    expected = RandomForestRegressor(n_estimators=5, random_state=42).fit(
        X[observed], y[observed], sample_weight=weight[observed],
    )

    np.testing.assert_allclose(masked.predict(X), expected.predict(X))


def test_unobserved_validation_rejected(censored):
    (X, _, y, censor), _ = censored

    for model in boosters(5):
        with pytest.raises(ValueError, match="No observed targets"):
            get_adapter(model).fit(
                model, X, y,
                censor=censor,
                loss=Loss(masked=True),
                validation=(X[:10], np.full(10, np.nan)),
            )


class TargetRecorder:
    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        return np.zeros((len(X), self.y_.shape[1]))


def test_partially_missing_targets_warn_and_pass_through():
    X, truth, *_ = make_censored(0, n=12)
    Y = np.c_[truth, -truth]
    Y[::3, 1] = np.nan
    Y[1] = np.nan
    model = TargetRecorder()

    with pytest.warns(RuntimeWarning, match="cannot mask individual targets"):
        get_adapter(model).fit(model, X, Y, loss=Loss(masked=True))

    np.testing.assert_array_equal(model.y_, np.delete(Y, 1, axis=0))


def test_masked_multitarget():
    xgb = pytest.importorskip("xgboost")
    cat = pytest.importorskip("catboost")
    X, truth, *_ = make_censored(0, n=300)
    Y = np.c_[truth, -truth]
    Y_masked = Y.copy()
    Y_masked[::3, 1] = np.nan
    Y_masked[::7, 0] = np.nan
    Y_masked[::21] = np.nan

    for model in (
        xgb.XGBRegressor(n_estimators=50, n_jobs=1, random_state=42),
        cat.CatBoostRegressor(
            iterations=50, loss_function="MultiRMSE",
            verbose=False, thread_count=1, random_seed=42,
        ),
    ):
        get_adapter(model).fit(
            model, X, Y_masked,
            loss=Loss(masked=True),
            validation=(X, Y_masked),
        )
        prediction = model.predict(X)

        assert prediction.shape == Y.shape
        assert np.mean(np.abs(prediction - Y)) < 1.


def test_lightgbm_rejects_multitarget():
    lgb = pytest.importorskip("lightgbm")
    X, truth, *_ = make_censored(0, n=50)
    model = lgb.LGBMRegressor()

    with pytest.raises(NotImplementedError, match="single target"):
        get_adapter(model).fit(
            model, X, np.c_[truth, -truth], loss=Loss(masked=True),
        )


def unsupported_model(family):
    if family == "catboost":
        cat = pytest.importorskip("catboost")
        return cat.CatBoostRegressor(iterations=5, verbose=False)
    return RandomForestRegressor(n_estimators=5)


@pytest.mark.parametrize("family", ["sklearn", "catboost"])
def test_censoring_unsupported(family):
    model = unsupported_model(family)
    X, _, y, censor = make_censored(0, n=20)

    with pytest.raises(NotImplementedError, match="Censored"):
        get_adapter(model).fit(model, X, y, censor=censor)


@pytest.mark.parametrize("family", ["sklearn", "catboost"])
def test_custom_loss_unsupported(family):
    model = unsupported_model(family)
    X, _, y, _ = make_censored(0, n=20)

    with pytest.raises(NotImplementedError, match="Custom loss"):
        get_adapter(model).fit(model, X, y, loss=Loss(CUSTOM_LOSS))


def box_config(**box):
    pytest.importorskip("xgboost")
    return {
        "model": {
            "kind": "ensemble",
            "base": {
                "class_path": "xgboost.XGBRegressor",
                "init_args": {
                    "n_estimators": 100,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "n_jobs": 1,
                },
            },
            "n_members": 2,
        },
        "pipeline": {},
        "features": ["x0", "x1"],
        "target": ["y"],
        **box,
    }


def make_censored_dataset(seed, n=300):
    X, truth, y, censor = make_censored(seed, n=n)
    y[::17] = np.nan
    return Dataset.from_dict({
        "x0": X[:, 0].tolist(),
        "x1": X[:, 1].tolist(),
        "y": y.tolist(),
        "qualifier": censor.tolist(),
    }), truth


@pytest.fixture
def censored_dataset():
    return make_censored_dataset(0)


def test_box_censored_masked_checkpoint(tmp_path, censored_dataset):
    data, truth = censored_dataset
    box = Box.from_config(
        box_config(censor="qualifier", loss={"masked": True})
    )
    box.fit(data, skip_eval=True)
    before = np.asarray(box.predict(data)["prediction:y"])

    assert abs(np.mean(before - truth)) < 0.3
    config = box.to_config()
    assert config["censor"] == ["qualifier"]
    assert config["loss"] == {"name": "squared_error", "masked": True}

    box.save(tmp_path / "box")
    restored = Box.load(tmp_path / "box")

    assert restored.censor == box.censor
    assert restored.loss == box.loss
    assert Box.from_config(restored.to_config()).to_config() == config
    np.testing.assert_allclose(
        np.asarray(restored.predict(data)["prediction:y"]),
        before,
    )


def test_box_censor_one_column_per_target():
    with pytest.raises(ValueError, match="one column per target"):
        Box.from_config(box_config(censor=["a", "b"]))


def test_box_without_loss_keeps_missing_targets_an_error(censored_dataset):
    data, _ = censored_dataset
    box = Box.from_config(box_config())

    assert box.censor is None and box.loss is None
    assert box.to_config()["loss"] is None
    with pytest.raises(Exception, match="(?i)label|nan"):
        box.fit(data, skip_eval=True)


def test_exact_observed():
    y_true = np.array([[1., 5.], [5., np.nan], [5., 2.]])
    y_pred = np.array([[2., 7.], [3., 0.], [7., 1.]])
    censor = np.array([[0, 1], [0, 0], [-1, 0]])

    np.testing.assert_array_equal(
        np.concatenate(_exact_observed(y_true, y_pred, censor, masked=True)),
        [1., 5., 2., 2., 3., 1.],
    )
    unfiltered = y_true[:, :1], y_pred[:, :1]
    for returned, given in zip(_exact_observed(*unfiltered), unfiltered):
        np.testing.assert_array_equal(returned, given)

    with pytest.raises(ValueError, match="No exact, observed targets"):
        _exact_observed(y_true[:1, 1:], y_pred[:1, 1:], censor[:1, 1:])


def exact_rmsd(predicted):
    y = np.asarray(predicted["y"], dtype=float)
    prediction = np.asarray(predicted["prediction:y"])
    exact = np.isfinite(y) & (np.asarray(predicted["qualifier"]) == 0)
    return np.sqrt(np.mean(np.square(prediction - y)[exact]))


@pytest.mark.parametrize("masked", [False, True])
def test_box_evaluate_scores_exact_targets(censored_dataset, masked):
    data, _ = censored_dataset
    config = box_config(censor="qualifier")
    if masked:
        config["loss"] = {"masked": True}
    else:
        data = data.filter(lambda row: np.isfinite(row["y"]))
    box = Box.from_config(config)
    box.fit(data, skip_eval=True)

    predicted, scores = box.evaluate(data, metrics=["rmsd", "pearson"])

    assert np.isfinite(list(scores.values())).all()
    assert scores["rmsd"] == pytest.approx(exact_rmsd(predicted))


def test_box_fit_validation_checkpoint_roundtrip(tmp_path):
    training, _ = make_censored_dataset(0)
    validation, _ = make_censored_dataset(1, n=150)
    config = box_config(censor="qualifier", loss={"masked": True})
    rounds = config["model"]["base"]["init_args"]["n_estimators"]
    box = Box.from_config(config)

    box.fit(
        training,
        validation=validation,
        use_validation_for_fit=True,
        early_stopping_rounds=10,
        skip_eval=False,
        metrics=["rmsd", "pearson"],
    )
    predicted, scores = box.evaluate(validation, metrics=["rmsd", "pearson"])

    assert all(
        0 <= member.best_iteration < rounds
        for member in box.model.members.values()
    )
    assert box._validation_score == pytest.approx(scores)
    assert np.isfinite(list(scores.values())).all()
    assert scores["rmsd"] == pytest.approx(exact_rmsd(predicted))

    box.save(tmp_path / "box")
    restored = Box.load(tmp_path / "box")
    restored_predicted, restored_scores = restored.evaluate(
        validation, metrics=["rmsd", "pearson"],
    )

    assert restored_scores == pytest.approx(scores)
    np.testing.assert_allclose(
        np.asarray(restored_predicted["prediction:y"]),
        np.asarray(predicted["prediction:y"]),
    )
