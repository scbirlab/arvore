diff --git a/arvore/box/base.py b/arvore/box/base.py
index 58d709e..7f33d9f 100644
--- a/arvore/box/base.py
+++ b/arvore/box/base.py
@@ -16,7 +16,6 @@ else:
     DataPipeline = Any
     Dataset, IterableDataset = Any, Any
 
-from ..config.instantiate import instantiate_object
 from ..metrics import DEFAULT_METRICS, METRICS_REGISTRY
 from ..models.ensemble import Ensemble
 from ..models.losses import Loss, resolve_loss
@@ -213,12 +212,13 @@ def _matrix(
     import numpy as np
     if len(dataset) == 0:
         raise ValueError(f"No rows in dataset: {dataset}")
+    n_rows = _data_len(dataset)
     blocks = ()
-    for col_group in column_groups:
-        group_blocks = []
-        for feature in col_group:
+    for columns in column_groups:
+        arrays = []
+        for column in columns:
             column_name = get_transformed_column(
-                feature, 
+                column, 
                 dataset, 
                 pipeline,
             )
@@ -227,14 +227,13 @@ def _matrix(
                 values = np.asarray(values)
                 if values.dtype == object:
                     values = np.stack(list(values))
-                values = values.reshape(_data_len(dataset), -1)
+                values = values.reshape(n_rows, -1)
             except (TypeError, ValueError) as error:
-                raise ValueError(f"Feature {feature!r} is not a regular array: {dataset[column_name]}") from error
+                raise ValueError(f"Column {column!r} is not a regular array: {dataset[column_name]}") from error
             if not np.issubdtype(values.dtype, np.number):
-                raise TypeError(f"Feature {feature!r} must be numeric, got {values.dtype}")
-            group_blocks.append(values)
-        group_blocks = np.concatenate(group_blocks, axis=1)
-        blocks += (group_blocks,)
+                raise TypeError(f"Column {column!r} must be numeric, got {values.dtype}")
+            arrays.append(values)
+        blocks += (np.concatenate(arrays, axis=1),)
     return blocks
     
 
@@ -353,7 +352,7 @@ class Box:
         if len(features) == 0:
             raise ValueError("features must be a nonempty list of column names")
         if len(target) == 0:
-            raise ValueError("features must be a nonempty list of column names")
+            raise ValueError("target must be a nonempty list of column names")
         if not all(isinstance(f, str) and f for f in features):
             raise ValueError("features must contain nonempty strings")
         if not all(isinstance(f, str) and f for f in target):
@@ -363,8 +362,6 @@ class Box:
             and callable(getattr(model, "predict", None))
         ):
             raise TypeError("model must implement .fit(X, y) and .predict(X) methods")
-        
-        from providence import DataResolver
 
         self.pipeline = _resolve_pipeline(
             pipeline,
@@ -375,9 +372,6 @@ class Box:
         self.target = target
         self.censor = censor
         self.loss = resolve_loss(loss)
-
-        from ..models.ensemble import Ensemble
-
         self._model_config_fallback = None
 
         if isinstance(self.model, Ensemble):
@@ -467,7 +461,6 @@ class Box:
 
     @property
     def model_config(self):
-        from ..models.ensemble import Ensemble
         if isinstance(self.model, Ensemble):
             return (
                 self.model.to_config()
@@ -502,7 +495,6 @@ class Box:
     ) -> "Box":
         """Construct a Box from JSON-compatible configuration."""
         from aspect.data import DataPipeline
-        from ..models.ensemble import Ensemble
 
         if isinstance(config, (str, Path)):
             if os.path.exists(config):
diff --git a/arvore/cli/evaluate.py b/arvore/cli/evaluate.py
index bf3f4a0..477e392 100644
--- a/arvore/cli/evaluate.py
+++ b/arvore/cli/evaluate.py
@@ -65,13 +65,6 @@ def _evaluate(args: Namespace) -> None:
     )
 
     # Convert NumPy scalar/array results to JSON-compatible values.
-    def serializable(value):
-        if hasattr(value, "tolist"):
-            return value.tolist()
-        raise TypeError(
-            f"Cannot serialize {type(value).__name__}"
-        )
-
     def serializable(value):
         if hasattr(value, "tolist"):
             value = value.tolist()
diff --git a/arvore/config/configs.py b/arvore/config/configs.py
index 9171e48..2341da7 100644
--- a/arvore/config/configs.py
+++ b/arvore/config/configs.py
@@ -2,7 +2,6 @@ from typing import ClassVar
 from collections.abc import Mapping
 from copy import deepcopy
 from dataclasses import dataclass, field, fields, asdict
-import os
 from math import isfinite
 
 
diff --git a/arvore/config/instantiate.py b/arvore/config/instantiate.py
index fcb5085..eeca2ff 100644
--- a/arvore/config/instantiate.py
+++ b/arvore/config/instantiate.py
@@ -1,6 +1,6 @@
-"""Interpret declarative PyTorch model specifications."""
+"""Interpret declarative model and experiment configurations."""
 
-from typing import TYPE_CHECKING, Any
+from typing import Any
 from collections.abc import Iterable, Mapping
 from copy import deepcopy
 from importlib.resources import files
@@ -84,24 +84,6 @@ def _load_override_file(path: Path) -> Mapping[str, Any]:
     return data
 
 
-def _load_override_file(path: Path) -> Mapping[str, Any]:
-    """Load a flat mapping of dotted overrides from JSON or YAML."""
-    import yaml
-
-    with path.open(encoding="utf-8") as handle:
-        if path.suffix.lower() == ".json":
-            data = json.load(handle)
-        else:
-            data = yaml.safe_load(handle)
-
-    if not isinstance(data, Mapping):
-        raise ValueError(
-            f"Override file {str(path)!r} must contain a mapping."
-        )
-
-    return data
-
-
 def apply_overrides(
     config: Mapping[str, ...],
     overrides: Iterable[str] | None = None,
diff --git a/arvore/metrics/base.py b/arvore/metrics/base.py
index 4c4c1c4..abce1a4 100644
--- a/arvore/metrics/base.py
+++ b/arvore/metrics/base.py
@@ -123,7 +123,7 @@ def mae(y_true, y_pred):
 def pearson_r(y_true, y_pred):
     import numpy as np
     y_pred, y_true = _normalize_dims(y_pred, y_true, broadcast=True)
-    return  np.corrcoef(y_pred, y_true)[0, 1]
+    return np.corrcoef(y_pred, y_true)[0, 1]
 
 
 @register_metric(name="spearman")
diff --git a/arvore/models/adapters/base.py b/arvore/models/adapters/base.py
index beb7f58..29991eb 100644
--- a/arvore/models/adapters/base.py
+++ b/arvore/models/adapters/base.py
@@ -1,6 +1,5 @@
 """Adapter registry and helpers shared by model families."""
 
-from collections.abc import Mapping
 from contextlib import contextmanager
 
 from carabiner.decorators import decorator_with_params
@@ -40,10 +39,6 @@ def register_adapter(
     return fn
 
 
-def _kwargs(fit_kwargs: Mapping[str, ...] | None):
-    return dict(fit_kwargs or {})
-
-
 def _require_validation(
     validation: Validation | None = None,
     rounds: int | None = None
diff --git a/arvore/models/adapters/catboost.py b/arvore/models/adapters/catboost.py
index d60b6cb..25611bc 100644
--- a/arvore/models/adapters/catboost.py
+++ b/arvore/models/adapters/catboost.py
@@ -8,7 +8,6 @@ from ..losses import Loss
 from ...typing import Estimator, Validation
 from .base import (
     _drop_masked,
-    _kwargs,
     _normalize_leaf_indices,
     _require_native_loss,
     _require_validation,
@@ -63,7 +62,7 @@ class CatBoostAdapter(SklearnAdapter):
         """Fit, training partially missing targets on ``MultiRMSEWithMissingValues``."""
         _require_validation(validation, early_stopping_rounds)
         _require_native_loss(estimator, loss, censor)
-        kw = _kwargs(fit_kwargs)
+        kw = dict(fit_kwargs or {})
 
         if "eval_set" in kw:
             raise ValueError("Use validation= instead of fit.eval_set")
diff --git a/arvore/models/adapters/lightgbm.py b/arvore/models/adapters/lightgbm.py
index e0c5e67..5fd4fdf 100644
--- a/arvore/models/adapters/lightgbm.py
+++ b/arvore/models/adapters/lightgbm.py
@@ -8,7 +8,6 @@ from ..losses import Loss
 from ...typing import Estimator, Validation
 from .base import (
     _drop_masked,
-    _kwargs,
     _normalize_leaf_indices,
     _require_validation,
     _temporary_params,
@@ -33,6 +32,10 @@ class LightGBMPredictor:
         )
 
 
+def _booster(model):
+    return model.booster if isinstance(model, LightGBMPredictor) else model.booster_
+
+
 def _lightgbm_has_eval_xy(estimator):
     """LightGBM >= 4.6 replaces eval_set with eval_X/eval_y."""
     from inspect import signature
@@ -64,14 +67,8 @@ class LightGBMAdapter(SklearnAdapter):
         return frozenset({"leaf_indices"})
 
     def leaf_indices(self, estimator, X):
-        booster = (
-            estimator.booster
-            if isinstance(estimator, LightGBMPredictor)
-            else estimator.booster_
-        )
-
         return _normalize_leaf_indices(
-            booster.predict(
+            _booster(estimator).predict(
                 X,
                 pred_leaf=True,
                 num_iteration=None,
@@ -97,12 +94,7 @@ class LightGBMAdapter(SklearnAdapter):
     def save(self, model, path, **kwargs):
         # Explicitly preserve the trained ensemble, including
         # its best-iteration prediction behaviour.
-        booster = (
-            model.booster
-            if isinstance(model, LightGBMPredictor)
-            else model.booster_
-        )
-        booster.save_model(str(path))
+        _booster(model).save_model(str(path))
 
     def load(self, path, *, trusted=(), **kwargs):
         import lightgbm as lgb
@@ -133,7 +125,7 @@ class LightGBMAdapter(SklearnAdapter):
         if censor is not None and loss is None:
             loss = Loss()
         _require_validation(validation, early_stopping_rounds)
-        kw = _kwargs(fit_kwargs)
+        kw = dict(fit_kwargs or {})
 
         if {"eval_set", "eval_X", "eval_y"} & kw.keys():
             raise ValueError(
diff --git a/arvore/models/adapters/persistence.py b/arvore/models/adapters/persistence.py
index 04a0a8e..2a25b16 100644
--- a/arvore/models/adapters/persistence.py
+++ b/arvore/models/adapters/persistence.py
@@ -18,7 +18,6 @@ def save_model(
     skip_save: bool = False,
     **kwargs
 ):
-
     path = Path(path)
     path.mkdir(parents=True, exist_ok=True)
 
@@ -27,20 +26,20 @@ def save_model(
         allow_unsafe=kwargs.get("allow_unsafe", False),
     )
     path = path / adapter.filename
-    if not skip_save:
-        adapter.save(model, path,  **kwargs)
     if skip_save:
         trainable = True
-    elif adapter.name == "ensemble":
-        from providence.io import load_json
-
-        manifest = load_json(path)
-        trainable = all(
-            entry["metadata"]["trainable"]
-            for entry in manifest["members"]
-        )
     else:
-        trainable = adapter.restored_trainable
+        adapter.save(model, path, **kwargs)
+        if adapter.name == "ensemble":
+            from providence.io import load_json
+
+            manifest = load_json(path)
+            trainable = all(
+                entry["metadata"]["trainable"]
+                for entry in manifest["members"]
+            )
+        else:
+            trainable = adapter.restored_trainable
 
     return ModelArtifact(
         adapter=adapter.name,
@@ -107,11 +106,9 @@ def load_model(
 
     adapter = get_adapter_by_name(name)
     filename = artifact.filename
+    # ModelArtifact sets filename and sha256 together, or neither.
     if filename is None:
-        if artifact.sha256 is None:
-            raise ValueError(f"Cannot load from checkpoint where mdoel svaing was skipped")
-        else:
-            raise FileNotFoundError(f"Checksum exists but filename is missing")
+        raise ValueError("Cannot load a model from a checkpoint saved with skip_save=True")
     if filename != adapter.filename:
         raise ValueError(
             f"Unexpected filename for adapter {name}: {filename}"
diff --git a/arvore/models/adapters/sklearn.py b/arvore/models/adapters/sklearn.py
index 26d3db9..08536dc 100644
--- a/arvore/models/adapters/sklearn.py
+++ b/arvore/models/adapters/sklearn.py
@@ -10,7 +10,6 @@ from ...typing import Estimator, Validation
 from .base import (
     _class_path,
     _drop_masked,
-    _kwargs,
     _normalize_leaf_indices,
     _require_native_loss,
     register_adapter,
@@ -32,12 +31,10 @@ def _array_digest(array) -> str:
 
 
 def _skops_digest(raw: bytes) -> str:
-    """Content identity of a skops archive, stable across save and load.
-
-    skops archives aren't byte-reproducible: entries are named by id(),
-    timestamped, and tree arrays carry uninitialized struct padding.
-    Hash the schema without ids, and arrays field by field.
-    """
+    """Content identity of a skops archive, stable across save and load."""
+    # skops archives aren't byte-reproducible: entries are named by id(),
+    # timestamped, and tree arrays carry uninitialized struct padding.
+    # Hash the schema without ids, and arrays field by field.
     import io
     import json
     import zipfile
@@ -146,7 +143,7 @@ class SklearnAdapter:
                 "early_stopping_rounds"
             )
         _require_native_loss(estimator, loss, censor)
-        kw = _kwargs(fit_kwargs)
+        kw = dict(fit_kwargs or {})
         X, y, _ = _drop_masked(loss, X, y, fit_kwargs=kw)
         if loss is not None and loss.requires_objective(y):
             import warnings
diff --git a/arvore/models/adapters/xgboost.py b/arvore/models/adapters/xgboost.py
index 957a05c..d4e5c6c 100644
--- a/arvore/models/adapters/xgboost.py
+++ b/arvore/models/adapters/xgboost.py
@@ -8,7 +8,6 @@ from ..losses import Loss
 from ...typing import Estimator, Validation
 from .base import (
     _drop_masked,
-    _kwargs,
     _normalize_leaf_indices,
     _require_validation,
     _temporary_params,
@@ -99,7 +98,7 @@ class XGBoostAdapter(SklearnAdapter):
         if censor is not None and loss is None:
             loss = Loss()
         _require_validation(validation, early_stopping_rounds)
-        kw = _kwargs(fit_kwargs)
+        kw = dict(fit_kwargs or {})
 
         if "eval_set" in kw:
             raise ValueError("Use validation= instead of fit.eval_set")
diff --git a/arvore/models/ensemble.py b/arvore/models/ensemble.py
index f7d473b..7442b0b 100644
--- a/arvore/models/ensemble.py
+++ b/arvore/models/ensemble.py
@@ -1,7 +1,6 @@
 """Composable regression ensembles."""
 
-from collections.abc import Callable, Mapping, Iterable
-from copy import deepcopy
+from collections.abc import Mapping, Iterable
 from math import isfinite
 from pathlib import Path
 import os
diff --git a/arvore/scorers/leaves.py b/arvore/scorers/leaves.py
index 3be688c..821fe6f 100644
--- a/arvore/scorers/leaves.py
+++ b/arvore/scorers/leaves.py
@@ -38,10 +38,10 @@ class LeafReference:
 
         scores = np.zeros(n_samples, dtype=float)
 
-        for t, (ids, counts) in enumerate(
+        for tree, (ids, counts) in enumerate(
             zip(self.ids, self.counts)
         ):
-            query = leaves[:, t]
+            query = leaves[:, tree]
             positions = np.searchsorted(ids, query)
 
             valid = positions < len(ids)
@@ -146,12 +146,12 @@ def _decode_leaf(archive, calibrated):
         ):
             raise ValueError("Invalid saved leaf-reference arrays")
         tree_ids = tuple(
-            np.asarray(ids[a:b], dtype=np.int64).copy()
-            for a, b in zip(offsets[:-1], offsets[1:])
+            np.asarray(ids[start:stop], dtype=np.int64).copy()
+            for start, stop in zip(offsets[:-1], offsets[1:])
         )
         tree_counts = tuple(
-            np.asarray(counts[a:b], dtype=np.int64).copy()
-            for a, b in zip(offsets[:-1], offsets[1:])
+            np.asarray(counts[start:stop], dtype=np.int64).copy()
+            for start, stop in zip(offsets[:-1], offsets[1:])
         )
         for one_ids, one_counts in zip(tree_ids, tree_counts):
             if (
@@ -217,9 +217,9 @@ class LeafReferenceBuilder:
                 "Number of trees changed between batches"
             )
 
-        for t, table in enumerate(self._counts):
+        for tree, table in enumerate(self._counts):
             ids, counts = np.unique(
-                leaves[:, t],
+                leaves[:, tree],
                 return_counts=True,
             )
 
diff --git a/arvore/scorers/persistence.py b/arvore/scorers/persistence.py
index 50e6344..fe0583d 100644
--- a/arvore/scorers/persistence.py
+++ b/arvore/scorers/persistence.py
@@ -26,11 +26,9 @@ def _sha256(path):
 
 
 def fitted_state_digest(checkpoint, config):
-    """Hash saved model, fitted preprocessing and Box config.
-
-    Exclude only the reference archive and the top-level manifest to avoid a
-    circular dependency. Hash symbolic link targets without following them.
-    """
+    """Hash saved model, fitted preprocessing and Box config."""
+    # Exclude only the reference archive and the top-level manifest, which
+    # depend on this digest. Hash symbolic link targets without following them.
     checkpoint = Path(checkpoint)
     digest = sha256(json.dumps(
         config, sort_keys=True, separators=(",", ":"), allow_nan=False,
diff --git a/arvore/scorers/reference.py b/arvore/scorers/reference.py
index 2efcb55..61da058 100644
--- a/arvore/scorers/reference.py
+++ b/arvore/scorers/reference.py
@@ -3,11 +3,6 @@
 from collections.abc import Mapping
 
 
-def training_data(box):
-    """Resolve the fitted training data retained by Box."""
-    return box._training_reference()
-
-
 def iter_batches(data, batch_size):
     """Re-iterable Dataset or column-mapping batches."""
     if batch_size < 1:
diff --git a/arvore/scorers/runner.py b/arvore/scorers/runner.py
index 4c3f6fa..9873c3f 100644
--- a/arvore/scorers/runner.py
+++ b/arvore/scorers/runner.py
@@ -72,21 +72,21 @@ class ScoringRunner:
         self.needs_leaves = False
 
         for name, scorer in self.scorers.items():
-            req = getattr(
+            requirements = getattr(
                 scorer,
                 "requirements",
                 Requirements(),
             )
 
-            if not isinstance(req, Requirements):
+            if not isinstance(requirements, Requirements):
                 raise TypeError(
                     f"Scorer {name!r} must declare Requirements"
                 )
 
             unsupported = (
-                set(req.outputs) - supported_outputs
+                set(requirements.outputs) - supported_outputs
             ) | (
-                set(req.capabilities) - supported_capabilities
+                set(requirements.capabilities) - supported_capabilities
             )
 
             if unsupported:
@@ -95,7 +95,7 @@ class ScoringRunner:
                     f"resources: {sorted(unsupported)}"
                 )
 
-            missing = set(req.columns) - columns
+            missing = set(requirements.columns) - columns
 
             if missing:
                 raise ValueError(
@@ -104,8 +104,8 @@ class ScoringRunner:
                 )
 
             needs_members = (
-                "member_predictions" in req.outputs
-                or "ensemble_members" in req.capabilities
+                "member_predictions" in requirements.outputs
+                or "ensemble_members" in requirements.capabilities
             )
 
             if needs_members and (
@@ -121,8 +121,8 @@ class ScoringRunner:
                 "leaf_indices",
             ):
                 requested = (
-                    capability in req.outputs
-                    or capability in req.capabilities
+                    capability in requirements.outputs
+                    or capability in requirements.capabilities
                 )
 
                 if not requested:
diff --git a/arvore/scorers/tanimoto.py b/arvore/scorers/tanimoto.py
index e5a5ef7..4242eab 100644
--- a/arvore/scorers/tanimoto.py
+++ b/arvore/scorers/tanimoto.py
@@ -132,7 +132,7 @@ class TanimotoReference:
         if query_chunk < 1 or reference_chunk < 1:
             raise ValueError("Chunk sizes must be positive")
 
-        queries, q_counts, _ = _encode_fingerprints(
+        queries, query_counts, _ = _encode_fingerprints(
             fingerprints, n_bits=self.n_bits
         )
         distances = np.empty(len(queries), dtype=np.float64)
@@ -146,9 +146,9 @@ class TanimotoReference:
         )
 
         for start in range(0, len(queries), query_chunk):
-            q = queries[start:start + query_chunk]
-            qc = q_counts[start:start + query_chunk]
-            width = len(q)
+            query = queries[start:start + query_chunk]
+            query_popcount = query_counts[start:start + query_chunk]
+            width = len(query)
 
             best_sim = np.full(
                 (width, n_neighbors), -np.inf, dtype=np.float64
@@ -160,14 +160,14 @@ class TanimotoReference:
             )
 
             for offset in range(0, len(self.packed), reference_chunk):
-                r = self.packed[offset:offset + reference_chunk]
-                rc = self.counts[offset:offset + reference_chunk]
+                reference = self.packed[offset:offset + reference_chunk]
+                reference_popcount = self.counts[offset:offset + reference_chunk]
 
                 intersection = np.bitwise_count(
-                    np.bitwise_and(q[:, None, :], r[None, :, :])
+                    np.bitwise_and(query[:, None, :], reference[None, :, :])
                 ).sum(axis=2, dtype=np.int32)
 
-                union = qc[:, None] + rc[None, :] - intersection
+                union = query_popcount[:, None] + reference_popcount[None, :] - intersection
                 similarity = np.divide(
                     intersection,
                     union,
@@ -176,7 +176,7 @@ class TanimotoReference:
                 )
 
                 index = np.broadcast_to(
-                    np.arange(offset, offset + len(r), dtype=np.int64),
+                    np.arange(offset, offset + len(reference), dtype=np.int64),
                     similarity.shape,
                 )
 
diff --git a/arvore/utils/plotting.py b/arvore/utils/plotting.py
index 7c6c430..12c74a2 100644
--- a/arvore/utils/plotting.py
+++ b/arvore/utils/plotting.py
@@ -2,7 +2,6 @@
 
 from carabiner.mpl import add_legend, grid, figsaver
 import pandas as pd
-import numpy as np
 
 
 def _plot_history(
 