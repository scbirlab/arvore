
"""Minimal fit adapters for sklearn-compatible estimators."""

from typing import Any, Protocol, TypeAlias
from collections.abc import Mapping
from dataclasses import dataclass

from numpy.typing import ArrayLike

from ..typing import Estimator

Validation: TypeAlias = tuple[ArrayLike, ArrayLike] | None


class FitAdapter(Protocol):
    def fit(
        self,
        estimator: Estimator,
        X: ArrayLike,
        y: ArrayLike,
        *,
        validation: Validation | None = None,
        fit_kwargs: Mapping[str, ...] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        ...


def _kwargs(fit_kwargs: Mapping[str, ...] | None):
    return dict(fit_kwargs or {})


def _require_validation(
    validation: Validation | None = None, 
    rounds: int | None = None
):
    if rounds is not None:
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("early_stopping_rounds must be a positive integer")
        if validation is None:
            raise ValueError("Early stopping requires a training validation set")


@dataclass
class SklearnAdapter:
    def fit(
        self, 
        estimator: Estimator, 
        X: ArrayLike, 
        y: ArrayLike, 
        *,
        validation: Validation | None = None,
        fit_kwargs: Mapping[str, ...] | None = None,
        early_stopping_rounds: int | None = None,
    ):
        if early_stopping_rounds is not None:
            raise ValueError(
                "Generic sklearn estimators do not support "
                "early_stopping_rounds"
            )
        estimator.fit(X, y, **_kwargs(fit_kwargs))


@dataclass
class XGBoostAdapter:
    def fit(
        self, 
        estimator: Estimator, 
        X: ArrayLike, 
        y: ArrayLike, 
        *,
        validation: Validation | None = None,
        fit_kwargs: Mapping[str, ...] | None = None,
        early_stopping_rounds=None,
    ):
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

        if validation is not None:
            kw["eval_set"] = [validation]

        estimator.fit(X, y, **kw)


@dataclass
class LightGBMAdapter:
    def fit(
        self, 
        estimator: Estimator, 
        X: ArrayLike, 
        y: ArrayLike, 
        *,
        validation: Validation | None = None,
        fit_kwargs: Mapping[str, ...] | None = None,
        early_stopping_rounds: int | None = None
    ):
        _require_validation(validation, early_stopping_rounds)
        kw = _kwargs(fit_kwargs)

        if "eval_set" in kw:
            raise ValueError("Use validation= instead of fit.eval_set")

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

        if validation is not None:
            kw["eval_set"] = [validation]

        if callbacks:
            kw["callbacks"] = callbacks

        estimator.fit(X, y, **kw)


@dataclass
class CatBoostAdapter:
    def fit(
        self, 
        estimator: Estimator, 
        X: ArrayLike, 
        y: ArrayLike, 
        *,
        validation: Validation | None = None,
        fit_kwargs: Mapping[str, ...] | None = None,
        early_stopping_rounds: int | None = None,
    ):
        _require_validation(validation, early_stopping_rounds)
        kw = _kwargs(fit_kwargs)

        if "eval_set" in kw:
            raise ValueError("Use validation= instead of fit.eval_set")

        if early_stopping_rounds is not None:
            if "early_stopping_rounds" in kw:
                raise ValueError("Duplicate early stopping configuration")
            kw["early_stopping_rounds"] = early_stopping_rounds

        if validation is not None:
            kw["eval_set"] = validation

        estimator.fit(X, y, **kw)


def get_fit_adapter(estimator: Estimator) -> FitAdapter:
    """Select a backend using the estimator's type.
    """
    from sys import modules

    if "xgboost" in modules:
        from xgboost import XGBModel
        if isinstance(estimator, XGBModel):
            return XGBoostAdapter()

    if "lightgbm" in modules:
        from lightgbm import LGBMModel
        if isinstance(estimator, LGBMModel):
            return LightGBMAdapter()

    if "catboost" in modules:
        from catboost import CatBoost
        if isinstance(estimator, CatBoost):
            return CatBoostAdapter()

    return SklearnAdapter()
