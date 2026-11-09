
"""Model serialization independent of Box and Aspect."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol



def _backend_version(codec_name: str) -> str:
    """Return the version of the installed codec backend."""
    from importlib import import_module

    module_name = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "skops": "skops",
        "joblib": "joblib",
    }[codec_name]

    module = import_module(module_name)
    return module.__version__



class ModelCodec(Protocol):
    name: str
    filename: str

    def save(self, model: Any, path: Path) -> None: ...

    def load(
        self, path: Path, *, trusted: tuple[str, ...] = ()
    ) -> Any: ...


def _class_path(obj):
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


class XGBoostCodec:
    name = "xgboost"
    filename = "model.ubj"

    def save(self, model, path):
        model.save_model(str(path))

    def load(self, path, *, trusted=()):
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(str(path))
        return model


class LightGBMPredictor:
    """Inference-only wrapper preserving the Box prediction API."""

    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(X)

    def fit(self, X, y, **kwargs):
        raise RuntimeError(
            "Native LightGBM checkpoints are inference-only. "
            "Create a new estimator to retrain."
        )


class LightGBMCodec:
    name = "lightgbm"
    filename = "model.txt"

    def save(self, model, path):
        # Explicitly preserve the trained ensemble, including
        # its best-iteration prediction behaviour.
        model.booster_.save_model(str(path))

    def load(self, path, *, trusted=()):
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        return LightGBMPredictor(booster)


class CatBoostCodec:
    name = "catboost"
    filename = "model.cbm"

    def save(self, model, path):
        model.save_model(str(path), format="cbm")

    def load(self, path, *, trusted=()):
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(str(path), format="cbm")
        return model


class SkopsCodec:
    name = "skops"
    filename = "model.skops"

    def save(self, model, path):
        import skops.io as sio
        sio.dump(model, path)

    def load(self, path, *, trusted=()):
        import skops.io as sio

        safe_types = {
            "sklearn.tree._tree.Tree",
        }

        unknown = sio.get_untrusted_types(file=path)
        disallowed = set(unknown) - safe_types - set(trusted)
        if disallowed:
            raise ValueError(
                "Untrusted model types in skops checkpoint: "
                f"{sorted(disallowed)}. Review these types "
                "before explicitly trusting them."
            )
        return sio.load(
            path, 
            trusted=list(trusted),
        )


class JoblibCodec:
    name = "joblib"
    filename = "model.joblib"

    def save(self, model, path):
        import joblib
        joblib.dump(model, path)

    def load(self, path, *, trusted=()):
        import joblib
        return joblib.load(path)


CODECS = {
    "xgboost": XGBoostCodec(),
    "lightgbm": LightGBMCodec(),
    "catboost": CatBoostCodec(),
    "skops": SkopsCodec(),
    "joblib": JoblibCodec(),
}

PACKAGES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "skops": "skops",
    "joblib": "joblib",
}


def get_model_codec(model, *, allow_unsafe=False):
    """Resolve a codec from the actual fitted estimator type."""
    from sys import modules

    if "xgboost" in modules:
        from xgboost import XGBRegressor
        if type(model) is XGBRegressor:
            return CODECS["xgboost"]

    if "lightgbm" in modules:
        from lightgbm import LGBMRegressor
        if type(model) is LGBMRegressor:
            return CODECS["lightgbm"]

    if "catboost" in modules:
        from catboost import CatBoostRegressor
        if type(model) is CatBoostRegressor:
            return CODECS["catboost"]

    # The normal sklearn serialization path.
    if type(model).__module__.startswith("sklearn."):
        return CODECS["skops"]

    if allow_unsafe:
        return CODECS["joblib"]

    raise TypeError(
        f"No supported codec for {_class_path(model)}. "
        "Use allow_unsafe=True to enable joblib."
    )


def save_model(
    model, directory, *, allow_unsafe=False
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    codec = get_model_codec(
        model, allow_unsafe=allow_unsafe
    )
    path = directory / codec.filename
    codec.save(model, path)

    return {
        "codec": codec.name,
        "filename": codec.filename,
        "class_path": _class_path(model),
        "version": _backend_version(PACKAGES[codec.name]),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "trainable": codec.name != "lightgbm",
    }


def load_model(
    directory,
    metadata,
    *,
    allow_unsafe=False,
    trusted_types=()
):
    directory = Path(directory)
    name = metadata["codec"]

    if name not in CODECS:
        raise ValueError(f"Unknown model codec: {name}")

    if name == "joblib" and not allow_unsafe:
        raise ValueError(
            "Joblib checkpoints require allow_unsafe=True"
        )

    codec = CODECS[name]
    filename = metadata["filename"]

    if filename != codec.filename:
        raise ValueError(
            f"Unexpected filename for codec {name}: {filename}"
        )

    path = directory / filename
    digest = sha256(path.read_bytes()).hexdigest()

    if digest != metadata["sha256"]:
        raise ValueError("Model checkpoint checksum mismatch")

    return codec.load(
        path, trusted=tuple(trusted_types)
    )
