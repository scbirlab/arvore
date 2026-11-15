"""Adapter for CatBoost regressors, persisted as native CBM files."""

from .base import (
    _normalize_leaf_indices,
    register_adapter,
)
from .sklearn import SklearnAdapter


@register_adapter(name="catboost", ext="cbm")
class CatBoostAdapter(SklearnAdapter):

    estimator_classes = ("catboost.CatBoostRegressor",)
    supports_early_stopping = True

    def capabilities(self, estimator):
        return frozenset({"leaf_indices"})

    def leaf_indices(self, estimator, X):
        return _normalize_leaf_indices(
            estimator.calc_leaf_indexes(X),
            len(X),
        )

    def save(self, model, path, **kwargs):
        model.save_model(str(path), format="cbm")

    def load(self, path, *, trusted=(), **kwargs):
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(str(path), format="cbm")
        return model

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
        if early_stopping_rounds is not None:
            if "early_stopping_rounds" in fit_kwargs:
                raise ValueError("Duplicate early stopping configuration")
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        if validation is not None:
            fit_kwargs["eval_set"] = validation[:2]

        # CatBoost freezes parameters once fitted, so this one persists.
        if loss is not None and loss.requires_objective(y):
            estimator.set_params(loss_function="MultiRMSEWithMissingValues")

        estimator.fit(X, y, **fit_kwargs)
