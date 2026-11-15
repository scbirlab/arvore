"""Adapter for scikit-learn estimators, persisted with skops."""

from collections.abc import Mapping
from hashlib import sha256

from numpy.typing import ArrayLike

from ..losses import Loss
from ...typing import Estimator, Validation
from .base import (
    _class_path,
    _drop_masked,
    _kwargs,
    _normalize_leaf_indices,
    _require_native_loss,
    register_adapter,
)


def _array_digest(array) -> str:
    """SHA-256 of array values, ignoring struct padding and layout."""
    import numpy as np

    digest = sha256(repr((
        array.dtype.descr if array.dtype.names else array.dtype.str,
        array.shape,
    )).encode("utf-8"))
    for field in array.dtype.names or (None,):
        values = array if field is None else array[field]
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _skops_digest(raw: bytes) -> str:
    """Content identity of a skops archive, stable across save and load.

    skops archives aren't byte-reproducible: entries are named by id(),
    timestamped, and tree arrays carry uninitialized struct padding.
    Hash the schema without ids, and arrays field by field.
    """
    import io
    import json
    import zipfile

    import numpy as np

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
        loss: Loss | None = None,
        censor: ArrayLike | None = None,
    ):
        """Fit on the estimator's own objective, dropping masked rows."""
        if early_stopping_rounds is not None:
            raise ValueError(
                "Generic sklearn estimators do not support "
                "early_stopping_rounds"
            )
        _require_native_loss(estimator, loss, censor)
        kw = _kwargs(fit_kwargs)
        X, y, _ = _drop_masked(loss, X, y, fit_kwargs=kw)
        if loss is not None and loss.requires_objective(y):
            import warnings

            warnings.warn(
                f"{_class_path(estimator)} cannot mask individual targets: "
                "only rows missing every target were dropped. Rows missing "
                "some targets are passed on as missing values, which most "
                "scikit-learn estimators reject.",
                RuntimeWarning,
                stacklevel=2,
            )
        estimator.fit(X, y, **kw)

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
        import numpy as np
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
