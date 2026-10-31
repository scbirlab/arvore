"""Interpret declarative PyTorch model specifications."""

from typing import TYPE_CHECKING, Any
from collections.abc import Mapping
from importlib.resources import files
import json
from pathlib import Path

if TYPE_CHECKING:
    from torch.nn import Module
    from .training import Trainer
else:
    Module = Any
    Trainer = Any


def resolve_class(
    class_path: str
):
    """Resolve an import path to a Python class."""

    from importlib import import_module

    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(
        import_module(module_name),
        class_name,
    )

def instantiate_object(
    config: Mapping[str, Any]
):
    """Instantiate one class_path/init_args object."""

    config = dict(config)
    class_path = config["class_path"]
    init_args = dict(config.get("init_args", {}))

    cls = resolve_class(class_path)
    return cls(**init_args)


def instantiate_model(
    config: Mapping[str, Any]
) -> Module:
    """Instantiate an nn.Module from a class_path/init_args specification.

    """
    from jsonargparse import ArgumentParser
    from torch import nn

    parser = ArgumentParser()
    parser.add_subclass_arguments(
        nn.Module,
        "model",
    )
    parsed = parser.parse_object({
        "model": dict(config),
    })
    instantiated = parser.instantiate(parsed)
    return instantiated.model


def instantiate_trainer(
    config: Mapping[str, Any],
) -> Trainer:
    """Instantiate a Trainer from declarative configuration."""

    from .training import Trainer

    config = dict(config)

    loss_config = config.pop("loss")
    if isinstance(loss_config, Mapping):
        loss = instantiate_object(loss_config)
    else:
        loss = loss_config

    new_kwargs = {}
    optimizer = config.pop("optimizer", None)
    if isinstance(optimizer, str):
        new_kwargs["optimizer"] = resolve_class(optimizer)

    scheduler = config.pop("scheduler", None)
    if isinstance(scheduler, str):
        new_kwargs["scheduler"] = resolve_class(scheduler)

    regularizer = config.pop("regularizer", None)
    if isinstance(regularizer, Mapping):
        new_kwargs["regularizer"] = instantiate_object(regularizer)
    elif isinstance(regularizer, (list, tuple)):
        _regularizer = []
        for item in regularizer:
            if isinstance(item, Mapping):
                item = instantiate_object(item)
            _regularizer.append(item)
        new_kwargs["regularizer"] = _regularizer
             
    kwargs = {
        "loss": loss,
        **config,
    } | new_kwargs

    return Trainer(**kwargs)


def instantiate_uncertainty(config):
    if config is None:
        return None

    return {
        name: instantiate_object(method)
        for name, method
        in config.items()
    }


def _deep_update(
    base: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively overlay one configuration mapping onto another."""

    out = dict(base)

    for key, value in update.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_update(
                out[key],
                value,
            )
        else:
            out[key] = value

    return out


def _parse_override_value(
    value: str,
):
    """Parse a CLI override using JSON semantics, falling back to string."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(
    config: Mapping[str, Any],
    overrides: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Apply dotted-path CLI overrides to configuration."""

    out = dict(config)

    for override in overrides or ():
        try:
            path, raw_value = override.split(
                "=",
                1,
            )
        except ValueError as error:
            raise ValueError(
                "Config overrides must have form "
                "'path.to.option=value', "
                f"but got {override!r}."
            ) from error

        keys = path.split(".")

        if any(not key for key in keys):
            raise ValueError(
                f"Invalid config override path {path!r}."
            )

        cursor = out

        for key in keys[:-1]:
            existing = cursor.get(key)

            if existing is None:
                cursor[key] = {}
            elif not isinstance(existing, Mapping):
                raise ValueError(
                    f"Cannot override {path!r}: "
                    f"{key!r} is not a mapping."
                )

            cursor = cursor[key]

        cursor[keys[-1]] = _parse_override_value(
            raw_value
        )

    return out


def model_aliases() -> tuple[str, ...]:
    """Return bundled model configuration aliases."""

    model_dir = files("duvidnn").joinpath(
        "data",
        "models",
    )

    return tuple(
        sorted(
            path.name.removesuffix(".json")
            for path in model_dir.iterdir()
            if path.name.endswith(".json")
        )
    )


def load_model_alias(
    name: str,
) -> dict[str, Any]:
    """Load a bundled model configuration."""

    path = (
        files("duvidnn")
        .joinpath(
            "data",
            "models",
            f"{name}.json",
        )
    )

    if not path.is_file():
        available = ", ".join(
            model_aliases()
        )

        raise ValueError(
            f"Unknown model alias {name!r}. "
            f"Available aliases: {available}"
        )

    return json.loads(
        path.read_text()
    )


def resolve_experiment_config(
    config: Mapping[str, Any],
    *,
    model: str | None = None,
    overrides: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Resolve model aliases and CLI overrides into one experiment config."""

    resolved = dict(config)

    if model is not None:
        resolved = _deep_update(
            resolved,
            {
                "box": {
                    "model": load_model_alias(
                        model
                    )
                }
            },
        )

    return apply_overrides(
        resolved,
        overrides,
    )