"""Model serialization independent of Box and Aspect."""

from typing import Any, Protocol
from collections.abc import Iterable
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path

from carabiner.decorators import decorator_with_params

CODECS = {}

PACKAGES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "skops": "skops",
    "joblib": "joblib",
    "ensemble": "treeton",
}

def _backend_version(codec_name: str) -> str:
    """Return the version of the installed codec backend."""
    from importlib import import_module

    module_name = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "skops": "skops",
        "joblib": "joblib",
        "treeton": "treeton",
    }[codec_name]

    module = import_module(module_name)
    return module.__version__



class ModelCodec(Protocol):
    name: str
    filename: str

    def save(self, model: Any, path: Path, **kwargs) -> None: 
        ...

    def load(
        self, 
        path: Path, 
        *, 
        trusted: tuple[str, ...] = (),
        **kwargs
    ) -> Any: 
        ...


@decorator_with_params
def register_codec(fn: ModelCodec, name: str, ext: str | None = None) -> ModelCodec:
    fn.name = name
    fn.filename = f"model.{ext or name}"
    CODECS[name] = fn() if isinstance(fn, type) else fn
    return fn


def _class_path(obj):
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


@register_codec(name="xgboost", ext="ubj")
class XGBoostCodec:

    def save(self, model, path, **kwargs):
        model.save_model(str(path))

    def load(self, path, *, trusted=(), **kwargs):
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


@register_codec(name="lightgbm", ext="txt")
class LightGBMCodec:

    def save(self, model, path, **kwargs):
        # Explicitly preserve the trained ensemble, including
        # its best-iteration prediction behaviour.
        model.booster_.save_model(str(path))

    def load(self, path, *, trusted=(), **kwargs):
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        return LightGBMPredictor(booster)


@register_codec(name="catboost", ext="cbm")
class CatBoostCodec:

    def save(self, model, path, **kwargs):
        model.save_model(str(path), format="cbm")

    def load(self, path, *, trusted=(), **kwargs):
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(str(path), format="cbm")
        return model


@register_codec(name="skops")
class SkopsCodec:

    def save(self, model, path, **kwargs):
        import skops.io as sio
        sio.dump(model, path)

    def load(self, path, *, trusted=(), **kwargs):
        import skops.io as sio

        unknown = sio.get_untrusted_types(file=path)
        disallowed = set(unknown) - set(trusted)
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


@register_codec(name="joblib", ext="joblib")
class JoblibCodec:

    def save(self, model, path, **kwargs):
        import joblib
        joblib.dump(model, path)

    def load(self, path, *, trusted=(), **kwargs):
        import joblib
        return joblib.load(path)


@register_codec(name="ensemble", ext="json")
class EnsembleCodec:
    """Composite codec delegating persistence to member codecs."""

    def save(
        self, 
        model,
        path: str | os.PathLike, 
        *, 
        allow_unsafe: bool = False,
        **kwargs
    ):
        return model.save(path, allow_unsafe=allow_unsafe)

    def load(
        self,
        path: str | os.PathLike,
        *,
        trusted: Iterable[str] = (),
        allow_unsafe: bool = False
    ):
        from .ensemble import Ensemble
        return Ensemble.load(
            path,
            trusted=trusted,
            allow_unsafe=allow_unsafe,
        )


def get_model_codec(model, *, allow_unsafe=False):
    """Resolve a codec from the actual fitted estimator type."""
    from sys import modules
    from .ensemble import Ensemble

    if isinstance(model, Ensemble):
        return CODECS["ensemble"]

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
    model, 
    path: str | os.PathLike, 
    *,
    skip_save: bool = False,
    **kwargs
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    codec = get_model_codec(
        model,
        allow_unsafe=kwargs.get("allow_unsafe", False),
    )
    path = path / codec.filename
    if not skip_save:
        codec.save(model, path,  **kwargs)
    if codec.name == "ensemble":
        from providence.io import load_json

        manifest = load_json(path)
        trainable = all(
            entry["metadata"]["trainable"]
            for entry in manifest["members"]
        )
    else:
        trainable = codec.name != "lightgbm"

    return {
        "codec": codec.name,
        "filename": codec.filename if not skip_save else None,
        "class_path": _class_path(model),
        "version": _backend_version(PACKAGES[codec.name]),
        "sha256": sha256(path.read_bytes()).hexdigest() if not skip_save else None,
        "trainable": trainable or skip_save,
    }


def load_model(
    path: str | os.PathLike,
    metadata,
    *,
    allow_unsafe: bool = False,
    trusted_types=(),
    **kwargs
):
    path = Path(path)
    name = metadata["codec"]

    if name not in CODECS:
        raise ValueError(f"Unknown model codec: {name}")

    if name == "joblib" and not allow_unsafe:
        raise ValueError(
            "Joblib checkpoints require allow_unsafe=True"
        )

    codec = CODECS[name]
    filename = metadata["filename"]
    if filename is None:
        if metadata["sha256"] is None:
            raise ValueError(f"Cannot load from checkpoint where mdoel svaing was skipped")
        else:
            raise FileNotFoundError(f"Checksum exists but filename is missing")
    if filename != codec.filename:
        raise ValueError(
            f"Unexpected filename for codec {name}: {filename}"
        )

    path = path / filename
    digest = sha256(path.read_bytes()).hexdigest()

    if digest != metadata["sha256"]:
        raise ValueError("Model checkpoint checksum mismatch")

    return codec.load(
        path, 
        trusted=tuple(trusted_types),
        allow_unsafe=allow_unsafe,
        **kwargs
    )
