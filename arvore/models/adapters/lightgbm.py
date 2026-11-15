"""Adapter for LightGBM estimators, persisted as native text boosters."""

from collections.abc import Mapping

from numpy.typing import ArrayLike

from ..losses import Loss
from ...typing import Estimator, Validation
from .base import (
    _drop_masked,
    _normalize_leaf_indices,
    _require_validation,
    _temporary_params,
    register_adapter,
)
from .sklearn import SklearnAdapter


class LightGBMPredictor:
    """Inference-only wrapper preserving the Box prediction API."""

    def __init__(self, booster):
        self.booster = booster

    def predict(self, X, **kwargs):
        return self.booster.predict(X, **kwargs)

    def fit(self, X, y, **kwargs):
        raise RuntimeError(
            "Native LightGBM checkpoints are inference-only. "
            "Create a new estimator to retrain."
        )


def _booster(model):
    return model.booster if isinstance(model, LightGBMPredictor) else model.booster_


def _lightgbm_has_eval_xy(estimator):
    """LightGBM >= 4.6 replaces eval_set with eval_X/eval_y."""
    from inspect import signature
    return "eval_X" in signature(type(estimator).fit).parameters


def _objective(loss, y, censor):
    lower, upper = loss.bounds(y, censor)

    def objective(y_true, y_pred, weight):
        return loss.gradients(y_pred, lower, upper, weight)

    return objective


def _metric(loss, y, censor):
    lower, upper = loss.bounds(y, censor)

    def metric(y_true, y_pred, weight):
        return loss.name, loss.mean(y_pred, lower, upper, weight), False

    return metric


@register_adapter(name="lightgbm", ext="txt")
class LightGBMAdapter(SklearnAdapter):

    def capabilities(self, estimator):
        return frozenset({"leaf_indices"})

    def leaf_indices(self, estimator, X):
        return _normalize_leaf_indices(
            _booster(estimator).predict(
                X,
                pred_leaf=True,
                num_iteration=None,
            ),
            len(X),
        )


    restored_trainable = False

    def adapter_name(self, estimator, *, allow_unsafe=False):
        from lightgbm import LGBMRegressor

        if isinstance(estimator, LightGBMPredictor):
            return self.name

        return super().adapter_name(
            estimator,
            allow_unsafe=allow_unsafe,
            _class=LGBMRegressor,
        )

    def save(self, model, path, **kwargs):
        # Explicitly preserve the trained ensemble, including
        # its best-iteration prediction behaviour.
        _booster(model).save_model(str(path))

    def load(self, path, *, trusted=(), **kwargs):
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        return LightGBMPredictor(booster)

    def fit(
        self,
        estimator: Estimator,
        X: ArrayLike,
        y: ArrayLike,
        *,
        validation: Validation | None = None,
        fit_kwargs: Mapping[str, ...] | None = None,
        early_stopping_rounds: int | None = None,
        loss: Loss | None = None,
        censor: ArrayLike | None = None,
    ):
        """Fit a single target, training on ``loss`` where it needs a custom objective.

        Censored targets default to ``Loss()``.
        """
        import numpy as np

        if np.ndim(y) > 1 and np.shape(y)[1] > 1:
            raise NotImplementedError("LightGBM supports a single target")
        if censor is not None and loss is None:
            loss = Loss()
        _require_validation(validation, early_stopping_rounds)
        kw = dict(fit_kwargs or {})

        if {"eval_set", "eval_X", "eval_y"} & kw.keys():
            raise ValueError(
                "Use validation= instead of fit.eval_set/eval_X/eval_y"
            )

        X, y, censor = _drop_masked(loss, X, y, censor, kw)
        if validation is not None:
            validation = _drop_masked(loss, *validation)
            X_val, y_val, censor_val = validation
            if _lightgbm_has_eval_xy(estimator):
                kw["eval_X"], kw["eval_y"] = (X_val,), (y_val,)
            else:
                kw["eval_set"] = [(X_val, y_val)]

        callbacks = list(kw.pop("callbacks", None) or [])

        if "early_stopping_rounds" in kw:
            raise ValueError(
                "Pass early_stopping_rounds directly to Box.fit()"
            )

        if early_stopping_rounds is not None:
            import lightgbm as lgb

            if estimator.get_params().get("boosting_type") == "dart":
                raise ValueError(
                    "LightGBM early stopping is not supported for DART"
                )

            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            )

        if callbacks:
            kw["callbacks"] = callbacks

        loss_params = {}
        if loss is not None and loss.requires_objective(y, censor):
            # Built-in metrics would score censoring bounds as exact targets.
            loss_params.update(objective=_objective(loss, y, censor), metric="None")
            if validation is not None:
                kw["eval_metric"] = _metric(loss, y_val, censor_val)

        with _temporary_params(estimator, **loss_params):
            estimator.fit(X, y, **kw)
