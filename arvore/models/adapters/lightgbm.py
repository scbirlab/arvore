"""Adapter for LightGBM estimators, persisted as native text boosters."""

import numpy as np

from .base import (
    _normalize_leaf_indices,
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

    estimator_classes = (
        "lightgbm.LGBMRegressor",
        "arvore.models.adapters.lightgbm.LightGBMPredictor",
    )
    restored_trainable = False
    supports_early_stopping = True
    supports_custom_objective = True

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


    def save(self, model, path, **kwargs):
        # Explicitly preserve the trained ensemble, including
        # its best-iteration prediction behaviour.
        _booster(model).save_model(str(path))

    def load(self, path, *, trusted=(), **kwargs):
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        return LightGBMPredictor(booster)

    def _fit(
        self,
        estimator,
        X,
        y,
        *,
        validation,
        fit_kwargs,
        early_stopping_rounds,
        loss,
        censor,
    ):
        if np.ndim(y) > 1 and np.shape(y)[1] > 1:
            raise NotImplementedError("LightGBM supports a single target")

        if validation is not None:
            X_val, y_val, censor_val = validation
            if _lightgbm_has_eval_xy(estimator):
                fit_kwargs["eval_X"], fit_kwargs["eval_y"] = (X_val,), (y_val,)
            else:
                fit_kwargs["eval_set"] = [(X_val, y_val)]

        callbacks = list(fit_kwargs.pop("callbacks", None) or [])

        if "early_stopping_rounds" in fit_kwargs:
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
            fit_kwargs["callbacks"] = callbacks

        loss_params = {}
        if loss is not None and loss.requires_objective(y, censor):
            # Built-in metrics would score censoring bounds as exact targets.
            loss_params.update(objective=_objective(loss, y, censor), metric="None")
            if validation is not None:
                fit_kwargs["eval_metric"] = _metric(loss, y_val, censor_val)

        with _temporary_params(estimator, **loss_params):
            estimator.fit(X, y, **fit_kwargs)
