"""Resolve adapters from estimators and saved codec names."""

from ...typing import Estimator, FitAdapter
from .base import ADAPTERS
from .catboost import CatBoostAdapter
from .ensemble import EnsembleAdapter
from .lightgbm import LightGBMAdapter, LightGBMPredictor
from .sklearn import SklearnAdapter
from .xgboost import XGBoostAdapter


def get_adapter(estimator: Estimator) -> FitAdapter:
    """Select a backend using the estimator's type.
    """
    from sys import modules
    from ..ensemble import Ensemble

    if isinstance(estimator, Ensemble):
        return EnsembleAdapter()

    if isinstance(estimator, LightGBMPredictor):
        return LightGBMAdapter()

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


def get_adapter_by_name(name: str):
    """Resolve a saved backend identifier."""
    try:
        return ADAPTERS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model codec: {name}"
        ) from error


def get_serialization_adapter(model, *, allow_unsafe=False):
    """Resolve persistence through the estimator's backend."""
    adapter = get_adapter(model)
    name = adapter.adapter_name(
        model,
        allow_unsafe=allow_unsafe,
    )
    return get_adapter_by_name(name)
