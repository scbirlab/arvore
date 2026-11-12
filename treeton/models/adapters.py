
"""Minimal fit adapters for sklearn-compatible estimators."""

from collections.abc import Mapping
from hashlib import sha256
import os
from pathlib import Path

from carabiner.decorators import decorator_with_params
from numpy.typing import ArrayLike

from ..config import ModelArtifact
from ..typing import Estimator, FitAdapter, Validation

ADAPTERS: dict[str, FitAdapter] = {}

PACKAGES: dict[str, str] = {}


def _class_path(obj):
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _backend_version(adapter_name: str) -> str:
    """Return the version of the installed codec backend."""
    from importlib import import_module

    module_name = PACKAGES[adapter_name]
    module = import_module(module_name)
    return module.__version__


@decorator_with_params
def register_adapter(
    fn: FitAdapter, 
    name: str, 
    ext: str | None = None,
    package: str | None = None
) -> FitAdapter:
    fn.name = name
    fn.filename = f"model.{ext or name}"
    ADAPTERS[name] = fn() if isinstance(fn, type) else fn
    PACKAGES[name] = package or name
    return fn


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


@register_adapter(name="skops")
class SklearnAdapter:

    name = None
    filename = None
    restored_trainable: bool = True

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

    def adapter_name(
        self, 
        estimator: Estimator,
        *,
        _class=None,
        allow_unsafe: bool = False
    ) -> str:
        """Choose a compatible serialization format."""

        if _class is not None and isinstance(_class, type):
            if isinstance(estimator, _class):
                return self.name

        # The normal sklearn serialization path.
        if type(estimator).__module__.startswith("sklearn."):
            return "skops"
        
        if allow_unsafe:
            return "joblib"
        else:
            raise TypeError(
                f"No supported codec for "
                f"{_class_path(estimator)}. "
                "Use allow_unsafe=True to enable arbitrary binary serialization."
            )

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

    def predict(
        self,
        estimator: Estimator, 
        X: ArrayLike, 
        **kwargs
    ):
        return estimator.predict(X, **kwargs)


@register_adapter(name="xgboost", ext="ubj")
class XGBoostAdapter(SklearnAdapter):

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


@register_adapter(name="lightgbm", ext="txt")
class LightGBMAdapter(SklearnAdapter):

    restored_trainable = False

    def adapter_name(self, estimator, *, allow_unsafe=False):
        from lightgbm import LGBMRegressor
        
        return super().adapter_name(
            estimator, 
            allow_unsafe=allow_unsafe,
            _class=LGBMRegressor,
        )

    def save(self, model, path, **kwargs):
        # Explicitly preserve the trained ensemble, including
        # its best-iteration prediction behaviour.
        model.booster_.save_model(str(path))

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


@register_adapter(name="catboost", ext="cbm")
class CatBoostAdapter(SklearnAdapter):

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


@register_adapter(name="ensemble", ext="json", package="treeton")
class EnsembleAdapter(SklearnAdapter):

    def adapter_name(self, estimator, *, allow_unsafe=False):
        return self.name

    def save(self, model, path, *, allow_unsafe=False, **kwargs):
        model.save(path, allow_unsafe=allow_unsafe)

    def load(
        self, path, *, trusted=(), allow_unsafe=False, **kwargs
    ):
        from .ensemble import Ensemble

        return Ensemble.load(
            path,
            trusted=trusted,
            allow_unsafe=allow_unsafe,
        )

    def fit(
        self,
        estimator,
        X,
        y,
        *,
        validation=None,
        fit_kwargs=None,
        early_stopping_rounds=None
    ):
        for member in estimator.members.values():
            adapter = get_adapter(member)
            adapter.fit(
                member,
                X,
                y,
                validation=validation,
                fit_kwargs=fit_kwargs,
                early_stopping_rounds=early_stopping_rounds,
            )

    
def get_adapter(estimator: Estimator) -> FitAdapter:
    """Select a backend using the estimator's type.
    """
    from sys import modules
    from .ensemble import Ensemble

    if isinstance(estimator, Ensemble):
        return EnsembleAdapter()

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


@register_adapter(name="joblib", ext="joblib")
class JoblibAdapter(SklearnAdapter):
    """Explicitly unsafe whole-object persistence."""

    def save(self, model, path, **kwargs):
        import joblib
        joblib.dump(model, path)

    def load(self, path, *, trusted=(), allow_unsafe=False, **kwargs):
        if not allow_unsafe:
            raise ValueError(
                "Joblib checkpoints require allow_unsafe=True"
            )
        import joblib
        return joblib.load(path)


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


def save_model(
    model, 
    path: str | os.PathLike, 
    *,
    skip_save: bool = False,
    **kwargs
):
    from ..config import ModelArtifact

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapter = get_serialization_adapter(
        model,
        allow_unsafe=kwargs.get("allow_unsafe", False),
    )
    path = path / adapter.filename
    if not skip_save:
        adapter.save(model, path,  **kwargs)
    if skip_save:
        trainable = True
    elif adapter.name == "ensemble":
        from providence.io import load_json

        manifest = load_json(path)
        trainable = all(
            entry["metadata"]["trainable"]
            for entry in manifest["members"]
        )
    else:
        trainable = adapter.restored_trainable

    return ModelArtifact(
        adapter=adapter.name,
        filename=adapter.filename if not skip_save else None,
        class_path=_class_path(model),
        version=_backend_version(adapter.name),
        sha256=(
            sha256(path.read_bytes()).hexdigest()
            if not skip_save else None
        ),
        trainable=trainable,
    ).to_dict()


def load_model(
    path: str | os.PathLike,
    artifact: ModelArtifact | Mapping,
    *,
    allow_unsafe: bool = False,
    trusted_types=(),
    **kwargs
):
    from ..config import ModelArtifact
    if not isinstance(artifact, ModelArtifact):
        artifact = ModelArtifact.from_dict(artifact)

    path = Path(path)
    adapter = get_adapter_by_name(artifact.adapter)
    name = metadata["adapter"]
    
    adapter = get_adapter_by_name(name)
    filename = artifact.filename
    if filename is None:
        if artifact.sha256 is None:
            raise ValueError(f"Cannot load from checkpoint where mdoel svaing was skipped")
        else:
            raise FileNotFoundError(f"Checksum exists but filename is missing")
    if filename != adapter.filename:
        raise ValueError(
            f"Unexpected filename for adapter {name}: {filename}"
        )

    path = path / filename
    digest = sha256(path.read_bytes()).hexdigest()

    if digest != artifact.sha256:
        raise ValueError("Model checkpoint checksum mismatch")

    return adapter.load(
        path, 
        trusted=tuple(trusted_types),
        allow_unsafe=allow_unsafe,
        **kwargs
    )
