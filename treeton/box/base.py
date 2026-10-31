"""Checkpointable composition of data processing and PyTorch models."""

from typing import Any, TypeAlias
from collections.abc import Callable, Mapping, Iterable
from copy import deepcopy
from pathlib import Path
import json
import os

from aspect.data import DataPipeline
from carabiner import print_err
import torch
from torch import nn
from torchmetrics import Metric, MetricCollection

from ..config import instantiate_model
from ..checkpoint_utils import save_json, load_json
from ..invoke import (
    DuvidaModel,
    ModelInvoker,
    TrainingDerivatives,
)
from ..invoke.duvida import DEFAULT_APPROXIMATOR
from ..mapping import ColumnMap
from ..training import Trainer
from ..utils.device import move_to_device
from ..uncertainty import normalize_uncertainty


CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pt"
MODEL_FILENAME = "model.pt"
PIPELINE_DIRNAME: str = "data"
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_PREDICTION_COLUMN: str = "prediction"
TRAINING_DERIVATIVES_FILENAME: str = "training_derivatives.pt"

PipelineLike: TypeAlias = DataPipeline | Mapping[str, Any] | str | Callable


def _json_copy(
    value: Mapping[Any, Any],
) -> dict[str, ...]:
    """Return a JSON-native copy, validating serializability."""
    return json.loads(
        json.dumps(value)
    )


def _resolve_pipeline(pipeline: PipelineLike | None = None) -> DataPipeline:
    """Resolve a pipeline specification."""
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


def _materialize_model(module: nn.Module) -> None:
    """Build modules before state restoration.

    Applies mainly to in-package models.

    """
    if hasattr(module, "_built"):
        if getattr(module, "_built") is False:
            build = getattr(module, "build")
            if callable(build):
                build()

    for child in module.children():
        _materialize_model(child)


def _model_device(model: nn.Module):
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _predict_map_batch(
    batch: Mapping[str, Any],
    *,
    box,
    pipeline: DataPipeline,
    device,
    uncertainty,
    uncertainty_state: Mapping[str, Any],
    prediction_column: str = DEFAULT_PREDICTION_COLUMN
) -> dict[str, ...]:
    runtime_batch = pipeline.collate(batch)
    runtime_batch = move_to_device(runtime_batch, device)
    inputs = box.invoker.inputs(runtime_batch)

    with torch.inference_mode():
        prediction = box.model(**inputs)
    result = {
        prediction_column: (
            prediction
            .detach()
            .cpu()
            .numpy()
        )
    }
    for name, method in uncertainty.items():
        value = method(
            box=box,
            batch=runtime_batch,
            prediction=prediction,
            state=uncertainty_state[name],
        )
        result[name] = (
            value
            .detach()
            .cpu()
            .numpy()
        )
    return result



def _load_training_derivatives(
    path: Path,
    map_location,
) -> TrainingDerivatives | None:
    filename = path / TRAINING_DERIVATIVES_FILENAME

    if not filename.exists():
        return None

    state = torch.load(
        filename,
        map_location=map_location,
        weights_only=True,
    )

    return TrainingDerivatives.from_state_dict(state)


def _validate_training_derivatives(
    model: nn.Module,
    derivatives: TrainingDerivatives | None,
) -> None:
    if derivatives is None:
        return

    parameters = dict(model.named_parameters())

    for label, tree in (
        ("fisher_score", derivatives.fisher_score),
        ("fisher_information", derivatives.fisher_information),
    ):
        if tree is None:
            continue

        if set(tree) != set(parameters):
            raise ValueError(
                f"Cached {label} parameter keys "
                "do not match the restored model."
            )
        for name, value in tree.items():
            if value.shape != parameters[name].shape:
                raise ValueError(
                    f"Cached {label} shape for "
                    f"{name!r} is {value.shape}, "
                    "but model parameter shape is "
                    f"{parameters[name].shape}."
                )


class Box:
    """Data processing, model invocation, and optional training.

    A Box constructed with :meth:`from_config` retains the declarative model
    specification used to construct its PyTorch module. Such Boxes are saved
    as configuration plus a ``state_dict``.

    A Box constructed directly from an arbitrary existing ``nn.Module`` has
    no generic reconstruction recipe. Such Boxes remain supported and fall
    back to whole-model PyTorch serialization.

    """

    def __init__(
        self,
        model: nn.Module,
        input_map: ColumnMap,
        pipeline: PipelineLike | None = None,
        # TODO: Make type hints more specific
        data: Any | None = None,
        trainer: Trainer | None = None,
        model_config: Mapping[str, Any] | None = None,
        training_derivatives: TrainingDerivatives | None = None,
    ):
        self.pipeline = _resolve_pipeline(pipeline)
        self.model = model
        self.input_map = input_map
        self.data = data
        self.trainer = trainer

        self.model_config = (
            _json_copy(model_config)
            if model_config is not None
            else None
        )
        self.invoker = ModelInvoker(
            model=self.model,
            input_map=self.input_map,
        )
        self.training_derivatives = training_derivatives


    @property
    def training_data_source(self):
        """Recorded provenance for the training data."""
        return getattr(
            self.pipeline,
            "data_source",
            None,
        )
    def verify_training_data_source(self) -> bool | None:
        """Verify the original training-data source when possible."""
        source = self.training_data_source
        if source is None:
            return None
        return source.verify()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str,
        *,
        cache_dir: str | None = None
    ) -> "Box":
        """Construct a Box from JSON-compatible configuration."""
        if isinstance(config, str):
            if os.path.exists(config):
                config = load_json(config)
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

        try:
            input_map_config = config["input_map"]
        except KeyError as error:
            raise ValueError(
                f"Box config requires an 'input_map'. Received {config=}."
            ) from error

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
            input_map=input_map,
            pipeline=pipeline,
            model_config=model_config,
        )

    def to_config(self, filename: str | None = None) -> dict[str, ...]:
        """Return the durable JSON configuration for this Box."""
        if not isinstance(self.pipeline, DataPipeline):
            raise TypeError(
                "Checkpoint serialization currently requires "
                "an aspect.DataPipeline. Arbitrary callable "
                "pipelines do not have a generic reconstruction "
                "representation."
            )

        config = {
            "model": (
                _json_copy(
                    self.model_config
                )
                if self.model_config is not None
                else None
            ),
            "pipeline": self.pipeline.to_config(),
            "input_map": self.input_map.to_config(),
        }
        if filename is not None:
            save_json(config, filename)
        return config

    def save(
        self, 
        path: str | os.PathLike,
        **pipeline_kwargs
    ) -> None:
        """Save configuration and model state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        config = self.to_config()
        pipeline_config = config.pop("pipeline")

        save_json(config, path / CONFIG_FILENAME)

        self.pipeline.save(
            path / PIPELINE_DIRNAME,
            **pipeline_kwargs,
        )

        if self.model_config is None:
            torch.save(
                self.model,
                path / MODEL_FILENAME,
            )
        else:
            _materialize_model(self.model)
            torch.save(
                self.model.state_dict(),
                path / WEIGHTS_FILENAME,
            )
        derivatives_path = path / TRAINING_DERIVATIVES_FILENAME
        if self.training_derivatives is not None:
            torch.save(
                self.training_derivatives.state_dict(),
                derivatives_path,
            )
        elif derivatives_path.exists():
            derivatives_path.unlink()

    

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        map_location: Any = "cpu",
        cache_dir: str | None = None
    ) -> "Box":
        """Restore a Box checkpoint."""
        path = Path(path)
        config = load_json(path / CONFIG_FILENAME)

        input_map = ColumnMap.from_config(config.get("input_map"))
        pipeline = DataPipeline.load(
            path / PIPELINE_DIRNAME,
            cache_dir=cache_dir,
        )
        pipeline_config = pipeline.to_config()
        config["pipeline"] = pipeline_config

        model_config = config.get("model")
        training_derivatives = _load_training_derivatives(
            path,
            map_location,
        )
        
        if model_config is not None:
            model = instantiate_model(model_config)
            _materialize_model(model)
            state_dict = torch.load(
                path / WEIGHTS_FILENAME,
                map_location=map_location,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            _validate_training_derivatives(
                model,
                training_derivatives,
            )
            return cls(
                model=model,
                input_map=input_map,
                pipeline=pipeline,
                model_config=model_config,
                training_derivatives=training_derivatives,
            )

        model = torch.load(
            path / MODEL_FILENAME,
            map_location=map_location,
            weights_only=False,
        )
        _validate_training_derivatives(
            model,
            training_derivatives,
        )
        return cls(
            model=model,
            input_map=input_map,
            pipeline=pipeline,
            training_derivatives=training_derivatives,
        )

    def prepare(self, data):
        """Apply the recorded Aspect pipeline."""
        return self.pipeline(data)

    def predict_batch(
        self,
        batch,
    ):
        """Predict from an already prepared runtime batch."""
        return self.invoker.predict(batch)

    def supervised_batch(
        self,
        batch,
    ):
        """Return predictions and target from an already prepared batch."""
        return self.invoker.supervised(batch)

    def _training_data(self):
        if not isinstance(self.pipeline, DataPipeline):
            raise TypeError(
                "Training data requires an "
                "aspect.DataPipeline."
            )

        if self.pipeline.data_out is not None:
            return self.pipeline.data_out

        if self.pipeline.data_in is not None:
            pipeline = self.pipeline.clone()

            return pipeline(self.pipeline.data_in)

        raise ValueError(
            "Training data require retained "
            "training data or enough source provenance "
            "to reconstruct them."
        )

    def fit(
        self,
        data,
        *,
        validation=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        validation_batch_size: int | None = None,
        dataloader_kwargs: Mapping[str, Any] | None = None,
        validation_dataloader_kwargs: Mapping[str, Any] | None = None
    ):
        """Fit the model using raw training and optional validation and test data."""

        if self.trainer is None:
            raise ValueError(
                "Box.fit() requires a trainer."
            )

        if not isinstance(self.pipeline, DataPipeline):
            raise TypeError(
                "Box.fit() requires an "
                "aspect.DataPipeline."
            )

        dataloader_kwargs = dict(dataloader_kwargs or {})
        train_data = self.pipeline(data)
        train_loader = self.pipeline.dataloader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            **dataloader_kwargs,
        )

        val_loader = None
        if validation is not None:
            validation_pipeline = self.pipeline.clone()
            validation_data = validation_pipeline(validation)

            validation_kwargs = dict(
                validation_dataloader_kwargs
                or dataloader_kwargs
            )

            val_loader = (
                validation_pipeline.dataloader(
                    validation_data,
                    batch_size=(
                        validation_batch_size
                        or batch_size
                    ),
                    shuffle=False,
                    **validation_kwargs,
                )
            )
        else:
            val_loader = None
        self.training_derivatives = None
        self.trainer.fit(
            model=self.model,
            invoker=self.invoker,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )
        return self

    def predict(
        self,
        data=None,
        pipeline: Mapping[str, Any] | DataPipeline | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        prediction_column: str = DEFAULT_PREDICTION_COLUMN,
        uncertainty=None,
        device=None,
        map_kwargs: Mapping[str, Any] | None = None
    ):
        """Predict over a dataset in bounded-memory batches."""

        map_kwargs = dict(map_kwargs or {})

        if pipeline is None:
            ref_pipeline = self.pipeline
        elif isinstance(pipeline, Mapping):
            ref_pipeline = DataPipeline(column_transforms=pipeline)
        elif not isinstance(pipeline, DataPipeline):
            raise ValueError(
                "If provided, `pipeline` must be a dict or aspect.DataPipeline "
                f"but was {type(pipeline)}: {pipeline}"
            )

        if not isinstance(ref_pipeline, DataPipeline):
            raise TypeError(
                "Box.predict() requires an "
                "aspect.DataPipeline."
            )

        if device is None:
            device = _model_device(self.model)
        else:
            device = torch.device(device)
            self.model.to(device)

        pipeline = ref_pipeline.clone()
        prepared = None
        if data is None:  # use training data
            prepared = self._training_data()
        if prepared is None:
            prepared = pipeline(data)

        self.model.eval()

        uncertainty = normalize_uncertainty(uncertainty)
        for method in uncertainty.values():
            prepare_candidates = getattr(
                method,
                "prepare_candidates",
                None,
            )
            if prepare_candidates is not None:
                prepared = prepare_candidates(prepared)

        uncertainty_state = {
            name: method.prepare(
                self,
                batch_size=batch_size,
                device=device,
            )
            for name, method
            in uncertainty.items()
        }

        for column in (
            prediction_column,
            *uncertainty,
        ):
            if column in prepared.column_names:
                prepared = prepared.remove_columns(column)

        return prepared.map(
            _predict_map_batch,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                "box": self,
                "pipeline": pipeline,
                "prediction_column": prediction_column,
                "uncertainty": uncertainty,
                "uncertainty_state": uncertainty_state,
                "device": device,
            },
            desc="Predicting",
            **map_kwargs,
        )

    def compute_training_derivatives(
        self,
        *,
        fisher_score: bool = False,
        fisher_information: Mapping[str, Any] | str | None = None,
        loss: Callable | None = None,
        loss_inputs: Mapping[str, str] | None = None,
        loss_reduction: str = "mean",
        batch_size: int = DEFAULT_BATCH_SIZE,
        device=None,
        dataloader_kwargs: Mapping[str, Any] | None = None,
    ):
        """Compute optional training-set derivatives for downstream UQ."""

        if (
            not fisher_score
            and fisher_information is None
        ):
            return self

        if loss is None:
            if self.trainer is None:
                raise ValueError(
                    "No loss was supplied and this Box "
                    "does not retain a Trainer. Pass `loss=` "
                    "explicitly."
                )

            loss = self.trainer.loss

            if loss_inputs is None:
                loss_inputs = self.trainer.loss_inputs

        data = self._training_data()

        if device is None:
            device = _model_device(self.model)
        else:
            device = torch.device(device)
            self.model.to(device)

        dataloader_kwargs = dict(
            dataloader_kwargs or {}
        )

        def make_loader():
            return self.pipeline.dataloader(
                data,
                batch_size=batch_size,
                shuffle=False,
                **dataloader_kwargs,
            )

        def prepare_batch(batch):
            return move_to_device(batch, device)

        functional = DuvidaModel(
            self.model,
            self.invoker,
        )

        self.model.eval()

        if self.training_derivatives is None:
            self.training_derivatives = TrainingDerivatives()

        if fisher_score:
            score, n_samples = functional.accumulate_fisher_score(
                make_loader(),
                loss,
                loss_inputs=loss_inputs,
                loss_reduction=loss_reduction,
                prepare_batch=prepare_batch,
            )
            self.training_derivatives.fisher_score = score
            self.training_derivatives.n_samples = n_samples
            self.training_derivatives.loss_reduction = loss_reduction

        if fisher_information is not None:
            if isinstance(
                fisher_information,
                str,
            ):
                fisher_information = {
                    "approximator": fisher_information,
                }
            else:
                fisher_information = dict(fisher_information)

            approximator = fisher_information.pop("approximator", DEFAULT_APPROXIMATOR)

            information, n_samples = (
                functional.accumulate_fisher_information(
                    make_loader(),
                    loss,
                    loss_inputs=loss_inputs,
                    loss_reduction=loss_reduction,
                    approximator=approximator,
                    prepare_batch=prepare_batch,
                    **fisher_information,
                )
            )

            self.training_derivatives.fisher_information = information
            self.training_derivatives.fisher_information_config = {
                "approximator": approximator,
                **fisher_information,
            }
            self.training_derivatives.n_samples = n_samples
            self.training_derivatives.loss_reduction = loss_reduction

        return self

    def evaluate(
        self,
        data,
        metrics: Mapping[str, Metric] | Iterable[Metric],
        *,
        pipeline: Mapping[str, Any] | DataPipeline | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device=None,
        metric_mask: Callable[[Mapping], torch.Tensor] | None = None,
        dataloader_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the model over a dataset using streaming metrics."""

        if pipeline is None:
            pipeline = self.pipeline.clone()
        elif isinstance(pipeline, Mapping):
            pipeline = DataPipeline(
                column_transforms=pipeline,
            )
        elif isinstance(pipeline, DataPipeline):
            pipeline = pipeline.clone()
        else:
            raise TypeError(
                "`pipeline` must be a mapping, "
                f"DataPipeline, or None, but was {type(pipeline)}: {pipeline}."
            )

        prepared = pipeline(data)

        dataloader = pipeline.dataloader(
            prepared,
            batch_size=batch_size,
            shuffle=False,
            **dict(dataloader_kwargs or {}),
        )

        if isinstance(metrics, Mapping):
            metrics = dict(metrics)
        else:
            metrics = {
                metric.__class__.__name__: metric
                for metric in metrics
            }

        collection = MetricCollection(deepcopy(metrics))

        if device is None:
            device = _model_device(self.model)

        collection = collection.to(device)

        was_training = self.model.training
        self.model.eval()

        try:
            with torch.inference_mode():
                for batch in dataloader:
                    batch = move_to_device(
                        batch,
                        device,
                    )
                    prediction, target = (
                        self.invoker
                        .supervised(batch)
                    )
                    if metric_mask is not None:
                        mask = metric_mask(batch)

                        prediction = prediction[mask]
                        target = target[mask]
                        if prediction.numel() == 0:
                            continue

                    collection.update(prediction, target)
        finally:
            self.model.train(was_training)

        return {
            name: (
                value.detach().cpu().item()
                if value.numel() == 1
                else value.detach().cpu()
            )
            for name, value
            in collection.compute().items()
        }
