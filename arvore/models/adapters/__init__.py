"""Fit, prediction and persistence adapters, one module per model family."""

from .base import ADAPTERS, PACKAGES, register_adapter
from .sklearn import SklearnAdapter, _skops_digest
from .xgboost import XGBoostAdapter
from .lightgbm import LightGBMAdapter, LightGBMPredictor
from .catboost import CatBoostAdapter
from .ensemble import EnsembleAdapter
from .joblib import JoblibAdapter
from .dispatch import get_adapter, get_adapter_by_name, get_serialization_adapter
from .persistence import load_model, model_digest, save_model

__all__ = [
    "ADAPTERS",
    "PACKAGES",
    "register_adapter",
    "SklearnAdapter",
    "XGBoostAdapter",
    "LightGBMAdapter",
    "LightGBMPredictor",
    "CatBoostAdapter",
    "EnsembleAdapter",
    "JoblibAdapter",
    "get_adapter",
    "get_adapter_by_name",
    "get_serialization_adapter",
    "load_model",
    "model_digest",
    "save_model",
]
