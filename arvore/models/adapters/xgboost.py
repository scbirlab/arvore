"""Adapter for XGBoost estimators, persisted as UBJSON."""

from collections.abc import Mapping

from numpy.typing import ArrayLike

from ..losses import Loss
from ...typing import Estimator, Validation
from .base import (
    _drop_masked,
    _kwargs,
    _normalize_leaf_indices,
    _require_validation,
    _temporary_params,
    register_adapter,
)
from .sklearn import SklearnAdapter


def _objective(loss, y, censor):
    lower, upper = loss.bounds(y, censor)

    def objective(y_true, y_pred, sample_weight=None):
        return loss.gradients(y_pred, lower, upper, sample_weight)

    return objective


def _metric(loss, y, censor):
    lower, upper = loss.bounds(y, censor)

    def metric(y_true, y_pred, sample_weight=None):
        return loss.mean(y_pred, lower, upper, sample_weight)

    metric.__name__ = loss.name
    return metric


def _finite_labels(y):
    """XGBoost rejects missing labels, even when a custom objective ignores them."""
    import numpy as np
    return np.where(np.isfinite(y), y, 0.)


@register_adapter(name="xgboost", ext="ubj")
class XGBoostAdapter(SklearnAdapter):

    def capabilities(self, estimator):
        if estimator.get_params().get("booster") == "gblinear":
            return frozenset()
        return frozenset({"leaf_indices"})

    def leaf_indices(self, estimator, X):
        if "leaf_indices" not in self.capabilities(estimator):
            raise NotImplementedError(
                "Linear XGBoost models have no leaves"
            )

        return _normalize_leaf_indices(
            estimator.apply(X),
            len(X),
        )


    def adapter_name(self, estimator, *, allow_unsafe=False):
        from xgboost import XGBRegressor
        return super().adapter_name(
            estimator,
            allow_unsafe=allow_unsafe,
            _class=XGBRegressor,
        )

    def save(self, model, path, **kwargs):
        model.save_model(str(path))

    def load(self, path, *, trusted=(), **kwargs):
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(str(path))
        return model

    def fit(
        self,
        estimator: Estimator,
        X: ArrayLike,
        y: ArrayLike,
        *,
        validation: Validation | None = None,
        fit_kwargs: Mapping[str, ...] | None = None,
        early_stopping_rounds=None,
        loss: Loss | None = None,
        censor: ArrayLike | None = None,
    ):
        """Fit, training on ``loss`` where it needs a custom objective.

        Censored targets default to ``Loss()``.
        """
        if censor is not None and loss is None:
            loss = Loss()
        _require_validation(validation, early_stopping_rounds)
        kw = _kwargs(fit_kwargs)

        if "eval_set" in kw:
            raise ValueError("Use validation= instead of fit.eval_set")

        # Modern XGBoost uses constructor parameters for these.
        params = {}
        for key in ("early_stopping_rounds", "callbacks"):
            if key in kw:
                params[key] = kw.pop(key)

        if early_stopping_rounds is not None:
            if "early_stopping_rounds" in params:
                raise ValueError("Duplicate early stopping configuration")
            params["early_stopping_rounds"] = early_stopping_rounds

        if params:
            estimator.set_params(**params)

        if estimator.get_params().get("early_stopping_rounds"):
            _require_validation(
                validation,
                estimator.get_params()["early_stopping_rounds"],
            )

        X, y, censor = _drop_masked(loss, X, y, censor, kw)
        if validation is not None:
            validation = _drop_masked(loss, *validation)

        loss_params = {}
        if loss is not None and loss.requires_objective(y, censor):
            loss_params["objective"] = _objective(loss, y, censor)
            if validation is not None:
                X_val, y_val, censor_val = validation
                loss_params["eval_metric"] = _metric(loss, y_val, censor_val)
                validation = X_val, _finite_labels(y_val), censor_val
            y = _finite_labels(y)

        if validation is not None:
            kw["eval_set"] = [validation[:2]]

        with _temporary_params(estimator, **loss_params):
            estimator.fit(X, y, **kw)
