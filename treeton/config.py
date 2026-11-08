"""Interpret declarative PyTorch model specifications."""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterable, Mapping
from copy import deepcopy
from importlib.resources import files
import json
from pathlib import Path


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
    config: Mapping[str, Any],
    class_key: str = "class_path",
    init_key: str = "init_args"
):
    """Instantiate one class_path/init_args object."""

    config = dict(config)
    class_path = config[class_key]
    init_args = dict(config.get(init_key, {}))
    cls = resolve_class(class_path)
    return cls(**init_args)


def _deep_update(
    base: Mapping[str, ...],
    update: Mapping[str, ...],
) -> dict[str, ...]:
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
    config: Mapping[str, ...],
    overrides: Iterable[str, ...] | None = None,
) -> dict[str, ...]:
    """Apply dotted-path CLI overrides to configuration."""
    result = deepcopy(dict(config))
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

        cursor = result
        for key in keys[:-1]:
            if key not in cursor:
                cursor[key] = {}
            elif not isinstance(cursor[key], Mapping):
                raise ValueError(
                    f"Cannot override {path!r}: "
                    f"{key!r} is not a mapping."
                )
            cursor = cursor[key]
        cursor[keys[-1]] = _parse_override_value(raw_value)

    return out


def model_aliases() -> tuple[str, ...]:
    """Return bundled model configuration aliases."""

    model_dir = files("treeton").joinpath(
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
        files("treeton")
        .joinpath(
            "data",
            "models",
            f"{name}.json",
        )
    )

    if not path.is_file():
        available = ", ".join(model_aliases())
        raise ValueError(
            f"Unknown model alias {name!r}. "
            f"Available aliases: {available}"
        )
    return json.loads(path.read_text())


def resolve_experiment_config(
    config: Mapping[str, ...],
    *,
    model: str | None = None,
    overrides: list[str] | tuple[str, ...] | None = None,
) -> dict[str, ...]:
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
