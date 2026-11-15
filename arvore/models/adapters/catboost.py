"""Adapter for CatBoost estimators, persisted as native CBM files."""

from collections.abc import Mapping

from numpy.typing import ArrayLike

from ..losses import Loss
from ...typing import Estimator, Validation
from .base import (
    _drop_masked,
    _kwargs,
    _normalize_leaf_indices,
    _require_native_loss,
    _require_validation,
    register_adapter,
)
from .sklearn import SklearnAdapter


@register_adapter(name="catboost", ext="cbm")
class CatBoostAdapter(SklearnAdapter):

    def capabilities(self, estimator):
        return frozenset({"leaf_indices"})

    def leaf_indices(self, estimator, X):
        return _normalize_leaf_indices(
            estimator.calc_leaf_indexes(X),
            len(X),
        )

    def adapter_name(self, estimator, *, allow_unsafe=False):
        from catboost import CatBoostRegressor
        
        return super().adapter_name(
            estimator, 
            allow_unsafe=allow_unsafe,
            _class=CatBoostRegressor,
        )

    def save(self, model, path, **kwargs):
        model.save_model(str(path), format="cbm")

    def load(self, path, *, trusted=(), **kwargs):
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(str(path), format="cbm")
        return model

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
        """Fit, training partially missing targets on ``MultiRMSEWithMissingValues``."""
        _require_validation(validation, early_stopping_rounds)
        _require_native_loss(estimator, loss, censor)
        kw = _kwargs(fit_kwargs)

        if "eval_set" in kw:
            raise ValueError("Use validation= instead of fit.eval_set")

        if early_stopping_rounds is not None:
            if "early_stopping_rounds" in kw:
                raise ValueError("Duplicate early stopping configuration")
            kw["early_stopping_rounds"] = early_stopping_rounds

        X, y, _ = _drop_masked(loss, X, y, fit_kwargs=kw)
        if validation is not None:
            kw["eval_set"] = _drop_masked(loss, *validation)[:2]

        # CatBoost freezes parameters once fitted, so this one persists.
        if loss is not None and loss.requires_objective(y):
            estimator.set_params(loss_function="MultiRMSEWithMissingValues")

        estimator.fit(X, y, **kw)
