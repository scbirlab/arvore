"""Checkpointable, reproducible composition of data processing and tree estimator."""

from typing import Any, TypeAlias
from collections.abc import Callable, Mapping, Iterable
from copy import deepcopy
from pathlib import Path
from providence.checkpoint_utils import save_json, load_json
import json
import os

from carabiner import print_err

from ..config import instantiate_model
from ..metrics import DEFAULT_METRICS, METRICS_REGISTRY

CONFIG_FILENAME = "config.json"
DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_PREDICTION_COLUMN: str = "prediction"


def _json_copy(
    value: Mapping[Any, ...],
) -> dict[str, ...]:
    """Return a JSON-native copy, validating serializability."""
    return json.loads(json.dumps(value))


def _resolve_pipeline(pipeline: PipelineLike | None = None) -> DataPipeline:
    """Resolve a pipeline specification."""
    from aspect.data import DataPipeline
    if isinstance(pipeline, DataPipeline):
        return pipeline

    if callable(pipeline):
        print_err(
            "[WARN] Only aspect.DataPipeline as a data pipeline allows serialization.", 
            f"You are using {pipeline}",
        )
        return pipeline
    
    if isinstance(pipeline, Mapping) or pipeline is None:
        return DataPipeline(
            column_transforms=dict(pipeline or {}),
        )

    if isinstance(pipeline, str):
        if os.path.exists(pipeline):
            return DataPipeline.from_file(pipeline)

        raise FileNotFoundError(
            "`pipeline` was a string, but no file found "
            f"called `{pipeline}`"
        )

    raise ValueError(
        "`pipeline` must be a dict, filename, callable, "
        "or aspect.DataPipeline, "
        f"but was {type(pipeline)}: {pipeline}"
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
    _metrics = {}
    for k, v in metrics.items():
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
        for i in enumerate(dataset):
            pass
        return i
    elif isinstance(dataset, Mapping):
        lens = {k: len(v) for k, v in dataset.items()}
        if len(set(lens.values())) == 1:
            return next(lens.values())
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
            try:
                values = np.asarray(dataset[column_name])
                if values.dtype == object:
                    values = np.stack(list(dataset[column_name]))
                values = values.reshape(_data_len(dataset), -1)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Feature {feature!r} is not a regular array") from error
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(f"Feature {feature!r} must be numeric, got {values.dtype}")
            group_blocks.append(values)
        np.concatenate(group_blocks, axis=1)
        blocks += (group_blocks,)
    return blocks
    

def _predict_map_batch(
    batch: Mapping[str, ...],
    *,
    model,
    column_groups: Iterable[Iterable[str]],
    pipeline: DataPipeline,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN
) -> dict[str, ...]:
    feature_cols, target_cols = column_groups
    x = _matrix(batch, pipeline, column_groups=feature_cols)
    predicted = np.asarray(model.predict(x))
    predicted = {
        f"{prediction_column}:{target}": p
        for target, p in zip(target_cols, predicted.T)
    }
    return result | predicted


class Box:
    """Data processing, and model training.

    A Box constructed with :meth:`from_config` retains the declarative model
    specification used to construct its model.

    A Box constructed directly from an arbitrary existing model has
    no generic reconstruction recipe. Such Boxes remain supported and fall
    back to whole-model serialization.

    """

    SCHEMA_VERSION: int = 1

    def __init__(
        self,
        model: Estimator,
        features: str | Iterable[str],
        target: str | Iterable[str],
        pipeline: PipelineLike | os.PathLike | str | None = None,
        # TODO: Make type hints more specific
        data: DataLike | None = None,
        model_config: Mapping[str, ...] | None = None
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
        
        self.pipeline = _resolve_pipeline(pipeline)
        self.model = model
        self.features = features
        self.data = data

        self.model_config = (
            _json_copy(model_config)
            if model_config is not None
            else None
        )
        self._validation_score = None


    @property
    def training_data_source(self):
        """Recorded provenance for the training data."""
        return getattr(self.pipeline.data, "data_source", None)

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
        model = instantiate_model(model_config)
        
        input_map = ColumnMap.from_config(input_map_config)

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
            path.mkdir(parents=True, exist_ok=True)
            save_json(config, path)
        return config

    def save(
        self, 
        path: str | os.PathLike,
        **pipeline_kwargs
    ) -> None:
        """Save configuration and model state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        must_pickle = hasattr(self.model, "save")
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "config": self.to_config(),
            "artefacts": {
                "model_text": "model.json" if not must_pickle else None,
                "model_binary": "model.joblib" if must_pickle else None,
                "training_data": "training-data.as",
            }
        }
        
        config = self.to_config()
        save_json(config, path / CONFIG_FILENAME)

        self.pipeline.save(
            path / manifest["artefacts"]["training_data"],
            **pipeline_kwargs,
        )
        if must_pickle:
            joblib.dump(
                self.model, 
                path / manifest["artefacts"]["model_binary"],
            )
        else:
            self.model.save(
                path / manifest["artefacts"]["model_text"],
            )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        cache_dir: str | None = None
    ) -> "Box":
        """Restore a Box checkpoint."""
        from aspect.data import DataPipeline
        path = Path(path)
        config = load_json(path / CONFIG_FILENAME)

        model_config = config["config"].get("model_config")
        features = config["config"]["features"]
        target = config["config"]["target"]

        pipeline = DataPipeline.load(
            path / config["artefacts"]["training_data"],
            cache_dir=cache_dir,
        )
        model_filename = config["artefacts"].get("model_text")
        from_pickle = model is None

        if from_pickle:
            model_filename = config["artefacts"].get("model_binary")
        
        if from_pickle:
            model = joblib.load(model_filename)
        else:
            if model_config is not None:
                model = instantiate_model(model_config)
                _materialize_model(model)
                model = model.load(model_filename)
        return cls(
            model=model,
            pipeline=pipeline,
            features=features,
            target=target,
            model_config=model_config,
        )

    def featurize(self, data, *, training: bool = False):
        """Apply the DataPipeline."""
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
        data,
        *,
        validation=None,
        skip_eval: bool = False,
        metrics: Iterable[str | Callable] | Mapping[str, str | Callable] | None = None,
        **kwargs
    ):
        """Fit the model using raw training and optional validation and test data."""
        from aspect.data import DataPipeline

        training_data, pipeline = self.featurize(data, training=True)
        train_args = (
            self._matrix(training_data, self.features, pipeline),
            self._matrix(training_data, self.target, pipeline),
        )
            
        if validation is not None:
            val_data, val_pipeline = self.featurize(validation, training=False)
            val_kwargs = {
                "eval": tuple(
                     *self._matrix(val_data, pipeline),
                )
            }
        else:
            val_kwargs = {}
        self.model.fit(
            *self._matrix(training_data, pipeline),
            **val_kwargs,
            **kwargs
        )
        return self

    def _pre_predict(
        self,
        data=None,
        pipeline: PipelineLike | None = None
    ):
        from aspect.data import DataPipeline

        if batch_size < 1:
            raise ValueError(f"batch_size must be positive ({batch_size})")
        map_kwargs = dict(map_kwargs or {})
        prepared = None
        if data is None:  # use training data
            data = self.pipeline.data_in.data
            if not self.pipeline.data_out is None:
                prepared = self.pipeline.data_out.data
        if prepared is None:
            prepared, pipeline = self.featurize(data, training=False)
        else:
            pipeline = self.pipeline.clone()
        return prepared.with_format(None), pipeline

    def predict(
        self,
        data=None,
        pipeline: PipelineLike | None = None,
        prediction_column: str = DEFAULT_PREDICTION_COLUMN,
        batch_size: int = DEFAULT_BATCH_SIZE,
        map_kwargs: Mapping[str, ...] | None = None
    ):
        """Predict over a dataset in bounded-memory batches."""
        prepared, pipeline = _pre_predict(
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
            },
            desc="Predicting",
            **map_kwargs,
        )

    def evaluate(
        self,
        data=None,
        *,
        metrics: Iterable[str | Callable] | Mapping[str, str | Callable] | None = None,
        metric_mask: Callable[[Mapping], ...] | None = None,
        **kwargs
    ) -> dict[str, ...]:
        """Evaluate the model over a dataset using streaming metrics."""
        metrics = _resolve_metrics(metrics)

        prepared, pipeline = _pre_predict(
            data=data,
            pipeline=pipeline
        )
        if mask is not None and callable(mask):
            prepared = prepared.filter(
                mask, 
                **({"desc": "Masking" if isinstance(prepared, Dataset) else {}}),
            )
        predicted = prepared.map(
            _predict_map_batch,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                "model": self.model,
                "pipeline": pipeline,
                "prediction_column": prediction_column,
                "column_groups": (self.features, self.target),
            },
            desc="Predicting",
            **map_kwargs,
        )
        predicted_cols = [col for col in predicted.column_names if col.startswith(prediction_column)]
        return predicted, {
            key: metric(
                _matrix(prepared, pipeline, predicted_cols),
                _matrix(prepared, pipeline, self.target),
            ) for key, metric in metrics.items()
        }
