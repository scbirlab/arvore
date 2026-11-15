"""Adapter for scikit-learn estimators, persisted with skops."""

from collections.abc import Mapping
from hashlib import sha256
from importlib import import_module
import sys
import warnings

import numpy as np
from numpy.typing import ArrayLike

from ..losses import Loss
from ...typing import Estimator, Validation
from .base import (
    _class_path,
    _drop_masked,
    _normalize_leaf_indices,
    _require_native_loss,
    _require_validation,
    register_adapter,
)


def _array_digest(array) -> str:
    """SHA-256 of array values, ignoring struct padding and layout."""
    digest = sha256(repr((
        array.dtype.descr if array.dtype.names else array.dtype.str,
        array.shape,
    )).encode("utf-8"))
    for field in array.dtype.names or (None,):
        values = array if field is None else array[field]
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _skops_digest(raw: bytes) -> str:
    """Content identity of a skops archive, stable across save and load."""
    # skops archives aren't byte-reproducible: entries are named by id(),
    # timestamped, and tree arrays carry uninitialized struct padding.
    # Hash the schema without ids, and arrays field by field.
    import io
    import json
    import zipfile

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:

        def canonical(node):
            if isinstance(node, list):
                return [canonical(item) for item in node]
            if not isinstance(node, dict):
                return node
            out = {}
            for key, value in node.items():
                if key == "__id__":
                    continue
                if key == "file" and node.get("type") == "numpy":
                    array = np.load(
                        io.BytesIO(archive.read(value)),
                        allow_pickle=False,
                    )
                    out[key] = _array_digest(array)
                elif key == "file":
                    out[key] = sha256(archive.read(value)).hexdigest()
                else:
                    out[key] = canonical(value)
            return out

        schema = canonical(json.loads(archive.read("schema.json")))

    return sha256(json.dumps(
        schema, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


@register_adapter(name="skops")
class SklearnAdapter:
    """Fit, predict and persist scikit-learn-compatible estimators.

    Model families subclass this, declaring the ``estimator_classes`` they
    handle and implementing ``_fit``. Estimators no family handles fall back
    to this adapter.
    """

    name = None
    filename = None
    estimator_classes: tuple[str, ...] = ()
    restored_trainable: bool = True
    supports_early_stopping: bool = False
    supports_custom_objective: bool = False

    def handles(self, estimator) -> bool:
        """Whether ``estimator`` is one of this family's classes."""
        for class_path in self.estimator_classes:
            module, _, name = class_path.rpartition(".")
            # Never import an optional backend just to rule it out.
            if module.partition(".")[0] not in sys.modules:
                continue
            if isinstance(estimator, getattr(import_module(module), name)):
                return True
        return False

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
        allow_unsafe: bool = False
    ) -> str:
        """Choose a compatible serialization format."""
        if self.handles(estimator):
            return self.name

        if type(estimator).__module__.startswith("sklearn."):
            return "skops"

        if allow_unsafe:
            return "joblib"
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
        loss: Loss | None = None,
        censor: ArrayLike | None = None,
    ):
        """Fit ``estimator``, masking and censoring targets as ``loss`` requires.

        Censored targets default to ``Loss()``. Families without custom
        objectives reject censored targets and custom losses.
        """
        if early_stopping_rounds is not None and not self.supports_early_stopping:
            raise ValueError(
                f"{_class_path(estimator)} does not support early_stopping_rounds"
            )
        _require_validation(validation, early_stopping_rounds)
        fit_kwargs = dict(fit_kwargs or {})
        if {"eval_set", "eval_X", "eval_y"} & fit_kwargs.keys():
            raise ValueError(
                "Use validation= instead of fit.eval_set/eval_X/eval_y"
            )
        if censor is not None and loss is None:
            loss = Loss()
        if not self.supports_custom_objective:
            _require_native_loss(estimator, loss, censor)

        X, y, censor = _drop_masked(loss, X, y, censor, fit_kwargs)
        if validation is not None:
            validation = _drop_masked(loss, *validation)

        self._fit(
            estimator,
            X,
            y,
            validation=validation,
            fit_kwargs=fit_kwargs,
            early_stopping_rounds=early_stopping_rounds,
            loss=loss,
            censor=censor,
        )

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
        if loss is not None and loss.requires_objective(y):
            warnings.warn(
                f"{_class_path(estimator)} cannot mask individual targets: "
                "only rows missing every target were dropped. Rows missing "
                "some targets are passed on as missing values, which most "
                "scikit-learn estimators reject.",
                RuntimeWarning,
                stacklevel=3,
            )
        estimator.fit(X, y, **fit_kwargs)

    def predict(
        self,
        estimator: Estimator,
        X: ArrayLike,
        **kwargs
    ):
        return estimator.predict(X, **kwargs)

    def capabilities(self, estimator) -> frozenset[str]:
        """Advertise supported prediction diagnostics."""
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            RandomForestRegressor,
        )

        if isinstance(
            estimator,
            (RandomForestRegressor, ExtraTreesRegressor),
        ):
            return frozenset({"tree_predictions", "leaf_indices"})

        return frozenset()

    def tree_predictions(self, estimator, X):
        """Return (samples, trees, targets) predictions."""
        from sklearn.utils.validation import check_is_fitted

        if "tree_predictions" not in self.capabilities(estimator):
            raise NotImplementedError(
                f"Tree predictions are unsupported for "
                f"{type(estimator).__name__}"
            )

        check_is_fitted(estimator)

        predictions = []

        for tree in estimator.estimators_:
            values = np.asarray(
                tree.predict(X),
                dtype=float,
            )

            if values.ndim == 1:
                values = values[:, None]

            predictions.append(values)

        return np.stack(predictions, axis=1)

    def leaf_indices(self, estimator, X):
        """Return terminal-node indices for each tree."""
        if "leaf_indices" not in self.capabilities(estimator):
            raise NotImplementedError(
                f"Leaf indices unsupported for "
                f"{type(estimator).__name__}"
            )

        return _normalize_leaf_indices(
            estimator.apply(X),
            len(X),
        )
