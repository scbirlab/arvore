"""Checkpointable, reproducible composition of data processing and tree estimator."""

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Mapping, Iterable
from copy import deepcopy
from pathlib import Path
import os

from providence.io import save_json, load_json

if TYPE_CHECKING:
    from aspect import DataPipeline
    from datasets import Dataset
else:
    DataPipeline = Any
    Dataset = Any

from ..models.ensemble import Ensemble
from ..models.losses import Loss, resolve_loss
from ..prediction import DEFAULT_PREDICTION_COLUMN, predict_map_batch
from ..tabular import is_dataset, matrix, resolve_pipeline, unwrap
from ..typing import Estimator, DataLike, PipelineLike
from .evaluation import exact_observed, resolve_metrics, targets_and_censor
from .identity import (
    box_fingerprint,
    operation_fingerprint,
    pipeline_identity,
    training_identity,
)

DEFAULT_BATCH_SIZE: int = 1024


class Box:
    """Data processing, and model training.

    The model is always held as an :class:`arvore.Ensemble`; a single
    estimator becomes a one-member ensemble. A Box constructed with
    :meth:`from_config` retains the declarative model specification used to
    construct its model. One constructed from existing estimators has no
    reconstruction recipe, but can still be checkpointed.

    ``censor`` names numeric columns, one per target, of censoring
    indicators: negative for left-censored values (the true value is at
    most the target), positive for right-censored (at least the target)
    and zero for exact values. Censored targets train on ``loss``, by
    default :class:`arvore.Loss`. A ``loss`` (name, mapping of
    :class:`arvore.Loss` fields, importable function or
    :class:`arvore.Loss`) can also mask missing targets. Both are
    checkpointed with the Box.

    """

    def __init__(
        self,
        model: Estimator,
        features: str | Iterable[str],
        target: str | Iterable[str],
        censor: str | Iterable[str] | None = None,
        loss: str | Mapping[str, ...] | Callable | Loss | None = None,
        pipeline: PipelineLike | os.PathLike | str | None = None,
        model_config: Mapping[str, ...] | None = None,
        cache_dir: str | None = None
    ):
        if isinstance(features, str):
            features = (features,)
        if isinstance(target, str):
            target = (target,)
        if isinstance(censor, str):
            censor = (censor,)
        features, target = tuple(features), tuple(target)
        if censor is not None:
            censor = tuple(censor)
            if len(censor) != len(target):
                raise ValueError(
                    f"censor must name one column per target, but {censor} "
                    f"does not match {target}"
                )
        if len(features) == 0:
            raise ValueError("features must be a nonempty list of column names")
        if len(target) == 0:
            raise ValueError("target must be a nonempty list of column names")
        if not all(isinstance(f, str) and f for f in features):
            raise ValueError("features must contain nonempty strings")
        if not all(isinstance(f, str) and f for f in target):
            raise ValueError("target must be a nonempty column name")
        if not (
            callable(getattr(model, "fit", None)) 
            and callable(getattr(model, "predict", None))
        ):
            raise TypeError("model must implement .fit(X, y) and .predict(X) methods")

        self.pipeline = resolve_pipeline(
            pipeline,
            cache_dir=cache_dir,
        )
        self.model = model if isinstance(model, Ensemble) else Ensemble(model)
        if model_config is not None:
            self.model._attach_config(model_config)
        self.features = features
        self.target = target
        self.censor = censor
        self.loss = resolve_loss(loss)
        self._references = None
        self._validation_score = None
        self._fingerprint = None
        self._training_identity = training_identity(self.pipeline)

    @property
    def references(self):
        """References for this fitted Box (shared by all scorer invocations).

        Call box.references.clear() after external estimator/data mutation.
        Do not fit and predict concurrently on a Box.
        """
        from ..scorers.references import ReferenceCache

        # Created on first use: a Box without prepared references stays
        # picklable, for example inside scorer closures hashed by Datasets.
        if self._references is None:
            self._references = ReferenceCache()
        return self._references

    @property
    def fingerprint(self) -> str:
        """Deterministic identity of the fitted Box.

        Built from the content digest of the fitted model, the preprocessing
        config, features, targets and the provenance of the training data.
        Stable across save and load. Computed once per fit or load.
        """
        if self._fingerprint is None:
            self._fingerprint = box_fingerprint(self)
        return self._fingerprint

    @property
    def model_config(self):
        return self.model.to_config() if self.model.has_config else None

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
                f"saved whole-model checkpoint. Received {config=}."
            )

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
            censor=config.get("censor"),
            loss=config.get("loss"),
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
            "model": self.model_config,
            "pipeline": self.pipeline.to_config(),
            "features": list(self.features),
            "target": list(self.target),
            "censor": list(self.censor) if self.censor is not None else None,
            "loss": self.loss.to_dict() if self.loss is not None else None,
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
        *,
        score=None,
        save_references: bool = True,
        **pipeline_kwargs,
    ) -> None:
        """Atomically checkpoint a Box and optionally its references.

        When save_references=True, prepare resources requested by score
        and persist all supported previously prepared resources.

        When False, neither prepare nor persist scorer references.
        Fitted model, pipeline and retained training data are still saved.
        Prediction itself never implicitly writes scorer references.
        """
        from .checkpoint import save

        save(
            self,
            path,
            allow_unsafe,
            score=score,
            save_references=save_references,
            **pipeline_kwargs,
        )

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
        from .checkpoint import load

        return load(
            cls,
            path,
            allow_unsafe,
            trusted_types=trusted_types,
            cache_dir=cache_dir,
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
        return unwrap(pipeline(data)), pipeline

    @property
    def _target_columns(self) -> tuple[tuple[str, ...], ...]:
        """Target columns, then any censoring indicator columns."""
        if self.censor is not None:
            return (self.target, self.censor)
        return (self.target,)

    def _matrix(
        self, 
        dataset: Dataset
    ):
        return matrix(
            dataset,
            column_groups=(self.features, *self._target_columns),
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
        self.references.clear()
        self._fingerprint = None

        training_data, pipeline = self.featurize(
            data, 
            training=True,
        )
        self._training_identity = training_identity(pipeline)
        X, *targets = self._matrix(training_data)
        y, censor = targets_and_censor(targets)
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
            validation_data, _ = self.featurize(
                validation,
                training=False,
            )

            X_val, *val_targets = self._matrix(validation_data)
            fit_validation = (X_val, *targets_and_censor(val_targets))

        self.model.fit(
            X,
            y,
            validation=fit_validation,
            early_stopping_rounds=early_stopping_rounds,
            loss=self.loss,
            censor=censor,
            **kwargs,
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

            if pipeline is None:
                # Featurized once, on first use, then reused.
                return self.training_data, self.pipeline.clone()

            data = self.pipeline.data_in

        working_pipeline = (
            self.pipeline.clone()
            if pipeline is None
            else resolve_pipeline(pipeline).clone()
        )

        prepared = unwrap(working_pipeline(data))
        return prepared.with_format(None), working_pipeline

    @property
    def training_data(self):
        """Training rows after preprocessing, retaining raw columns."""
        if self.pipeline.data_out is not None:
            data = self.pipeline.data_out

        elif self.pipeline.data_in is not None:
            data = self.pipeline(self.pipeline.data_in)

        else:
            raise ValueError(
                "Reference scoring requires training data "
                "in the Aspect pipeline"
            )
        data = unwrap(data)
        return (
            data.with_format(None)
            if hasattr(data, "with_format")
            else data
        )

    def predict(
        self,
        data=None,
        pipeline: PipelineLike | None = None,
        prediction_column: str = DEFAULT_PREDICTION_COLUMN,
        batch_size: int = DEFAULT_BATCH_SIZE,
        map_kwargs: Mapping[str, ...] | None = None,
        *,
        score=None,
        **kwargs,
    ):
        """Predict in batches with optional composable scorers."""
        from ..scorers import resolve_scorers
        from ..scorers.runner import ScoringRunner

        map_kwargs = dict(map_kwargs or {})

        if batch_size < 1:
            raise ValueError(
                f"batch_size must be positive ({batch_size})"
            )

        scorers = resolve_scorers(score)

        if (
            scorers
            and map_kwargs.get("num_proc", 1) not in (None, 1)
        ):
            raise ValueError(
                "Multiprocess scoring is not supported yet"
            )

        prepared, pipeline = self._pre_predict(
            data=data,
            pipeline=pipeline,
        )

        scoring = None

        if scorers:
            scoring = ScoringRunner(
                box=self,
                scorers=scorers,
            )

            prepared = scoring.prepare_candidates(prepared)
            scoring.validate(prepared)
            scoring.prepare(batch_size=batch_size)

        if is_dataset(prepared):
            map_kwargs.setdefault(
                "new_fingerprint",
                operation_fingerprint(
                    self,
                    prepared,
                    "predict",
                    pipeline=pipeline_identity(pipeline),
                    prediction_column=prediction_column,
                    kwargs=kwargs,
                    scorers=scoring.scorers if scoring else None,
                ),
            )
        result = prepared.map(
            predict_map_batch,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                "model": self.model,
                "prediction_column": prediction_column,
                "column_groups": (
                    self.features,
                    self.target,
                ),
                "kwargs": kwargs,
                "scoring": scoring,
            },
            desc="Predicting",
            **map_kwargs,
        )

        if scoring is not None:
            scoring.finalize(result)

        return result

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
        """Return predicted dataset and evaluation scores.

        Metrics score exact targets only, excluding censored targets and,
        under a masked ``loss``, missing ones. Training, including early
        stopping, still uses censored targets through the loss. When any
        target is excluded, several targets are pooled into one array, so
        each target is weighted by its number of exact values, and
        per-target averaging is lost. Raises ``ValueError`` when no exact
        target remains, as for an entirely censored validation set.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive: {batch_size}")

        metric_fns = resolve_metrics(metrics)
        prepared, working_pipeline = self._pre_predict(
            data=data,
            pipeline=pipeline,
        )

        if metric_mask is not None:
            prepared = prepared.filter(metric_mask)
        if is_dataset(prepared):
            map_kwargs.setdefault(
                "new_fingerprint",
                operation_fingerprint(
                    self,
                    prepared,
                    "evaluate",
                    pipeline=pipeline_identity(working_pipeline),
                    prediction_column=prediction_column,
                ),
            )
        predicted = prepared.map(
            predict_map_batch,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                "model": self.model,
                "column_groups": (self.features, self.target),
                "prediction_column": prediction_column,
            },
            desc="Predicting",
            **map_kwargs,
        )

        prediction_cols = tuple(
            f"{prediction_column}:{target}"
            for target in self.target
        )

        (y_pred,) = matrix(
            predicted,
            (prediction_cols,),
        )
        y_true, *censor = matrix(
            predicted,
            self._target_columns,
        )

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: "
                f"{y_pred.shape} versus {y_true.shape}"
            )

        if y_true.shape[1] == 1:
            y_pred = y_pred[:, 0]
        y_true, censor = targets_and_censor((y_true, *censor))

        y_true, y_pred = exact_observed(
            y_true,
            y_pred,
            censor,
            masked=self.loss is not None and self.loss.masked,
        )

        scores = {
            name: fn(y_true, y_pred)
            for name, fn in metric_fns.items()
        }

        return predicted, scores
    