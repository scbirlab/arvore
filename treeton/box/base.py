"""Checkpointable, reproducible composition of data processing and tree estimator."""

from typing import TYPE_CHECKING, Any, TypeAlias
from collections.abc import Callable, Mapping, Iterable
from copy import deepcopy
from pathlib import Path
import json
import os

from carabiner import print_err
from providence.io import save_json, load_json

if TYPE_CHECKING:
    from aspect import DataPipeline
    from datasets import Dataset, IterableDataset
else:
    DataPipeline = Any
    Dataset, IterableDataset = Any, Any

from ..config import instantiate_object
from ..metrics import DEFAULT_METRICS, METRICS_REGISTRY
from ..typing import Estimator, DataLike, PipelineLike

CONFIG_FILENAME = "config.json"
DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_PREDICTION_COLUMN: str = "prediction"


def _json_copy(
    value: Mapping[Any, ...],
) -> dict[str, ...]:
    """Return a JSON-native copy, validating serializability."""
    return json.loads(json.dumps(value))


def _resolve_pipeline(
    pipeline: PipelineLike | None = None,
    *,
    cache_dir: str | None = None,
) -> DataPipeline:
    """Resolve an Aspect pipeline or its declarative configuration."""
    from aspect import DataPipeline

    if isinstance(pipeline, DataPipeline):
        return pipeline

    if pipeline is None:
        pipeline = {}

    if isinstance(pipeline, os.PathLike):
        pipeline = os.fspath(pipeline)

    if isinstance(pipeline, (str, Mapping)):
        return DataPipeline.from_config(
            pipeline,
            cache_dir=cache_dir,
        )

    raise TypeError(
        "pipeline must be an Aspect DataPipeline, "
        "a configuration mapping, or a JSON filename; "
        f"received {type(pipeline).__name__}"
    )


def _resolve_metrics(
    metrics: Iterable[str | Callable] | Mapping[str, str | Callable] | None = None
):
    if metrics is None:
        metrics = DEFAULT_METRICS
    if isinstance(metrics, (tuple, list)):
        new_metrics = {}
        for m in metrics:
            if callable(m):
                _m = str(m.__class__.__name__)
            elif isinstance(m, str):
                _m = m
            else:
                raise TypeError(f"Metrics must be callable or str, but was {type(m)}: {m}")
            new_metrics[_m] = m
    elif isinstance(metrics, Mapping):
        new_metrics = dict(metrics)
    else:
        raise TypeError(f"Metrics must be mapping or list, but was {type(m)}: {m}")
    _metrics = {}
    for k, v in new_metrics.items():
        if isinstance(v, str):
            _v = METRICS_REGISTRY.get(v)
        elif callable(v):
            _v = v
        else:
            raise TypeError(
                f"Metrics must be callable or supported str, but was {type(v)}: {v}\n"
                f"Supported: {METRICS_REGISTRY}"
            ) 
        _metrics[k] = _v
    return _metrics


def _col_names(dataset: Dataset | IterableDataset | Mapping) -> list[str]:
    from datasets import Dataset, IterableDataset
    if isinstance(dataset, (Dataset, IterableDataset)):
        return dataset.column_names
    elif isinstance(dataset, Mapping):
        return list(dataset)
    else:
        raise ValueError(f"Can't get column names for {type(dataset)}: {dataset}")
    

def _data_len(dataset: Dataset | IterableDataset | Mapping) -> int:
    from datasets import Dataset, IterableDataset
    if isinstance(dataset, Dataset):
        return len(dataset)
    elif isinstance(dataset, IterableDataset):
        for i, _ in enumerate(dataset):
            pass
        return i + 1
    elif isinstance(dataset, Mapping):
        lens = {k: len(v) for k, v in dataset.items()}
        if len(set(lens.values())) == 1:
            return next(iter(lens.values()))
        else:
            raise AttributeError(
                f"Column lengths of Mapping are inconsistent: {lens}"
            )
    else:
        raise ValueError(f"Can't get number of rows for {type(dataset)}: {dataset}")


def get_transformed_column(
    column: str, 
    dataset: Dataset | IterableDataset | Mapping, 
    pipeline: DataPipeline
) -> str:
    transforms = pipeline.column_transforms.get(column, ())
    column_names = _col_names(dataset)
    if len(transforms) > 0:
        generated_column = transforms[-1].output_column
        if generated_column in column_names:
            return generated_column
    if column in column_names:
        return column
    raise KeyError(
        f"Column or Aspect transform {column} unavailable. "
        f"Available columns: {column_names}"
    )


def _matrix(
    dataset: Dataset | Mapping, 
    pipeline: DataPipeline,
    column_groups: Iterable[Iterable[str]]
):
    import numpy as np
    if len(dataset) == 0:
        raise ValueError(f"No rows in dataset: {dataset}")
    blocks = ()
    for col_group in column_groups:
        group_blocks = []
        for feature in col_group:
            column_name = get_transformed_column(
                feature, 
                dataset, 
                pipeline,
            )
            values = dataset[column_name]
            try:
                values = np.asarray(values)
                if values.dtype == object:
                    values = np.stack(list(values))
                values = values.reshape(_data_len(dataset), -1)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Feature {feature!r} is not a regular array: {dataset[column_name]}") from error
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(f"Feature {feature!r} must be numeric, got {values.dtype}")
            group_blocks.append(values)
        group_blocks = np.concatenate(group_blocks, axis=1)
        blocks += (group_blocks,)
    return blocks
    

def _predict_map_batch(
    batch: Mapping[str, ...],
    *,
    model,
    column_groups: Iterable[Iterable[str]],
    pipeline: DataPipeline,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    kwargs: Mapping | None = None
) -> dict[str, ...]:
    import numpy as np
    from ..models.adapters import get_adapter

    feature_cols, target_cols = column_groups
    kwargs = kwargs or {}
    X, *_ = _matrix(batch, pipeline, column_groups=(feature_cols,))
    y_pred = np.asarray(get_adapter(model).predict(model, X, **kwargs))
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    expected_shape = (len(X), len(target_cols))
    if y_pred.shape != expected_shape:
        raise ValueError(
            f"Expected predictions with shape {expected_shape}, "
            f"received {y_pred.shape}"
        )

    predicted = {
        f"{prediction_column}:{target}": p
        for target, p in zip(target_cols, y_pred.T)
    }
    return batch | predicted


class Box:
    """Data processing, and model training.

    A Box constructed with :meth:`from_config` retains the declarative model
    specification used to construct its model.

    A Box constructed directly from an arbitrary existing model has
    no generic reconstruction recipe. Such Boxes remain supported and fall
    back to whole-model serialization.

    """

    SCHEMA_VERSION: int = 2

    def __init__(
        self,
        model: Estimator,
        features: str | Iterable[str],
        target: str | Iterable[str],
        pipeline: PipelineLike | os.PathLike | str | None = None,
        # TODO: Make type hints more specific
        model_config: Mapping[str, ...] | None = None,
        cache_dir: str | None = None
    ):
        if isinstance(features, str):
            features = (features,)
        if isinstance(target, str):
            target = (target,)
        features, target = tuple(features), tuple(target)
        if len(features) == 0:
            raise ValueError("features must be a nonempty list of column names")
        if len(target) == 0:
            raise ValueError("features must be a nonempty list of column names")
        if not all(isinstance(f, str) and f for f in features):
            raise ValueError("features must contain nonempty strings")
        if not all(isinstance(f, str) and f for f in target):
            raise ValueError("target must be a nonempty column name")
        if not (
            callable(getattr(model, "fit", None)) 
            and callable(getattr(model, "predict", None))
        ):
            raise TypeError("model must implement .fit(X, y) and .predict(X) methods")
        
        from providence import DataResolver

        self.pipeline = _resolve_pipeline(
            pipeline,
            cache_dir=cache_dir,
        )
        self.model = model
        self.features = features
        self.target = target

        self.model_config = (
            _json_copy(model_config)
            if model_config is not None
            else None
        )
        self._validation_score = None


    @property
    def training_data_source(self):
        """Recorded provenance for the training data."""
        return getattr(self.pipeline.data_in, "data_source", None)

    def verify_training_data_source(self) -> bool | None:
        """Verify the original training-data source when possible."""
        source = self.training_data_source
        if source is None:
            return None
        return source.verify()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, ...] | os.PathLike | str,
        *,
        cache_dir: str | None = None
    ) -> "Box":
        """Construct a Box from JSON-compatible configuration."""
        from aspect.data import DataPipeline
        from ..models.ensemble import Ensemble

        if isinstance(config, (str, Path)):
            if os.path.exists(config):
                config = load_json(str(config))
            else:
                raise FileNotFoundError(
                    "`config` was string, but filename "
                    f"called `{config}` not found."
                )

        config = deepcopy(dict(config))
        try:
            model_config = config["model"]
        except KeyError as error:
            raise ValueError(
                f"Box config requires a 'model' specification. Received {config=}"
            ) from error

        if model_config is None:
            raise ValueError(
                f"Box.from_config() requires a model specification. "
                "Opaque models can only be reconstructed from a "
                "saved whole-model checkpoint. Received {config=}."
            )

        model_config = _json_copy(model_config)
        model = Ensemble.from_config(model_config)

        pipeline_config = config.get("pipeline", {})
        pipeline = DataPipeline.from_config(
            pipeline_config,
            cache_dir=cache_dir,
        )
        return cls(
            model=model,
            features=config.get("features", []),
            target=config.get("target", []),
            pipeline=pipeline,
            model_config=model_config,
        )

    def to_config(
        self, 
        path: str | os.PathLike | None = None
    ) -> dict[str, ...]:
        """Return the durable JSON configuration for this Box."""
        from aspect.data import DataPipeline
        if not isinstance(self.pipeline, DataPipeline):
            raise TypeError(
                "Checkpoint serialization currently requires "
                "an aspect.DataPipeline. Arbitrary callable "
                "pipelines do not have a generic reconstruction "
                "representation."
            )

        config = {
            "schema_version": self.SCHEMA_VERSION,
            "model_config": (
                _json_copy(self.model_config)
                if self.model_config is not None
                else None
            ),
            "pipeline": self.pipeline.to_config(),
            "features": list(self.features),
            "target": list(self.target),
        }
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_json(config, path)
        return config

    def save(
        self, 
        path: str | os.PathLike,
        allow_unsafe: bool = False,
        **pipeline_kwargs
    ) -> None:
        """Save configuration and model state."""
        from ..models.codecs import save_model
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_metadata = save_model(
            self.model,
            path,
            allow_unsafe=allow_unsafe,
        )
        
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "config": self.to_config(),
            "model": model_metadata,
            "artefacts": {
                "training_data": "training-data.as",
            }
        }
        self.pipeline.save(
            path / manifest["artefacts"]["training_data"],
            **pipeline_kwargs,
        )
        save_json(manifest, path / CONFIG_FILENAME)
        return None

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        allow_unsafe: bool = False,
        *,
        trusted_types: Iterable[str] | None = None,
        cache_dir: str | None = None
    ) -> "Box":
        """Restore a Box checkpoint."""
        from aspect.data import DataPipeline
        from ..models.codecs import load_model
        import joblib

        path = Path(path)
        manifest = load_json(path / CONFIG_FILENAME)
        schema = manifest["schema_version"]
        if trusted_types is None:
            trusted_types = ()
        if schema == 1:
            if not allow_unsafe:
                raise ValueError(
                    "Legacy schema-1 checkpoints use joblib. "
                    "Pass allow_unsafe=True only for trusted files."
                )
            model = joblib.load(
                path / manifest["artefacts"]["model_binary"]
            )

        else:
            if schema != cls.SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported Treeton checkpoint schema: "
                    f"{manifest['schema_version']}, but package is schema version {cls.SCHEMA_VERSION}"
                )
            model = load_model(
                path,
                manifest["model"],
                allow_unsafe=allow_unsafe,
                trusted_types=trusted_types,
            )
        artefacts = manifest["artefacts"]
        config = manifest["config"]
        pipeline = DataPipeline.load(
            path / artefacts["training_data"],
            cache_dir=cache_dir,
        )
        return cls(
            model=model,
            pipeline=pipeline,
            features=config["features"],
            target=config["target"],
            model_config=config.get("model_config"),
        )

    def featurize(self, data, *, training: bool = False):
        """Apply the DataPipeline."""
        from aspect import DataPipeline
        if not isinstance(self.pipeline, DataPipeline):
            raise TypeError(
                "Box.featurize() requires an "
                f"aspect.DataPipeline. Found {self.pipeline}"
            )
        pipeline = self.pipeline if training else self.pipeline.clone()
        return self._unwrap(pipeline(data)), pipeline

    @staticmethod
    def _unwrap(data):
        from providence import DataResolver
        return data.data if isinstance(data, DataResolver) else data

    def _matrix(
        self, 
        dataset: Dataset, 
        pipeline: DataPipeline
    ):
        return _matrix(
            dataset=dataset,
            pipeline=pipeline,
            column_groups=(self.features, self.target),
        )

    def fit(
        self,
        data: DataLike,
        *,
        validation=None,
        skip_eval: bool = False,
        metrics: Iterable[str | Callable] | Mapping[str, str | Callable] | None = None,
        use_validation_for_fit: bool = False,
        early_stopping_rounds: int | None = None,
        **kwargs
    ):
        """Fit the model using raw training and optional validation and test data."""
        from ..models import get_adapter

        training_data, pipeline = self.featurize(
            data, 
            training=True,
        )
        X, y = self._matrix(training_data, pipeline)
        # sklearn regressors generally expect a 1D single target.
        if y.shape[1] == 1:
            y = y[:, 0]
        if early_stopping_rounds is not None and not use_validation_for_fit:
            raise ValueError(
                "early_stopping_rounds requires "
                "use_validation_for_fit=True"
            )

        fit_validation = None

        if use_validation_for_fit:
            if validation is None:
                raise ValueError(
                    "use_validation_for_fit=True requires validation data"
                )

            # Clone the fitted Aspect pipeline for validation.
            validation_data, validation_pipeline = self.featurize(
                validation,
                training=False,
            )

            X_val, y_val = self._matrix(
                validation_data,
                validation_pipeline,
            )

            if y_val.shape[1] == 1:
                y_val = y_val[:, 0]

            fit_validation = (X_val, y_val)

        adapter = get_adapter(self.model)
        adapter.fit(
            self.model,
            X,
            y,
            validation=fit_validation,
            fit_kwargs=kwargs,
            early_stopping_rounds=early_stopping_rounds,
        )

        self._validation_score = None

        if validation is not None and not skip_eval:
            _, self._validation_score = self.evaluate(
                validation,
                metrics=metrics,
            )

        return self

    def _pre_predict(
        self,
        data=None,
        pipeline: PipelineLike | None = None,
    ):
        if data is None:
            if self.pipeline.data_in is None:
                raise ValueError("No training data or prediction data available")

            if pipeline is None and self.pipeline.data_out is not None:
                return (
                    self._unwrap(self.pipeline.data_out).with_format(None),
                    self.pipeline.clone(),
                )

            data = self.pipeline.data_in

        working_pipeline = (
            self.pipeline.clone()
            if pipeline is None
            else _resolve_pipeline(pipeline).clone()
        )

        prepared = self._unwrap(working_pipeline(data))
        return prepared.with_format(None), working_pipeline

    def predict(
        self,
        data=None,
        pipeline: PipelineLike | None = None,
        prediction_column: str = DEFAULT_PREDICTION_COLUMN,
        batch_size: int = DEFAULT_BATCH_SIZE,
        map_kwargs: Mapping[str, ...] | None = None,
        **kwargs
    ):
        """Predict over a dataset in bounded-memory batches."""
        map_kwargs = dict(map_kwargs or {})
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive ({batch_size})")
        prepared, pipeline = self._pre_predict(
            data=data,
            pipeline=pipeline
        )
        return prepared.map(
            _predict_map_batch,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                "model": self.model,
                "pipeline": pipeline,
                "prediction_column": prediction_column,
                "column_groups": (self.features, self.target),
                "kwargs": kwargs,
            },
            desc="Predicting",
            **map_kwargs,
        )

    def evaluate(
        self,
        data=None,
        pipeline: PipelineLike | None = None,
        *,
        metrics=None,
        metric_mask: Callable | None = None,
        prediction_column: str = DEFAULT_PREDICTION_COLUMN,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **map_kwargs,
    ):
        """Return predicted dataset and evaluation scores."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive: {batch_size}")

        metric_fns = _resolve_metrics(metrics)
        prepared, working_pipeline = self._pre_predict(
            data=data,
            pipeline=pipeline,
        )

        if metric_mask is not None:
            prepared = prepared.filter(metric_mask)

        predicted = prepared.map(
            _predict_map_batch,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                "model": self.model,
                "column_groups": (self.features, self.target),
                "pipeline": working_pipeline,
                "prediction_column": prediction_column,
            },
            desc="Predicting",
            **map_kwargs,
        )

        prediction_cols = tuple(
            f"{prediction_column}:{target}"
            for target in self.target
        )

        (y_pred,) = _matrix(
            predicted,
            working_pipeline,
            (prediction_cols,),
        )
        (y_true,) = _matrix(
            predicted,
            working_pipeline,
            (self.target,),
        )

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: "
                f"{y_pred.shape} versus {y_true.shape}"
            )

        if y_true.shape[1] == 1:
            y_pred = y_pred[:, 0]
            y_true = y_true[:, 0]

        scores = {
            name: fn(y_pred, y_true)
            for name, fn in metric_fns.items()
        }

        return predicted, scores
    