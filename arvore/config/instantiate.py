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


def _load_override_file(path: Path) -> Mapping[str, Any]:
    """Load a flat mapping of dotted overrides from JSON or YAML."""
    import yaml

    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError(
            f"Override file {str(path)!r} must contain a mapping."
        )

    return data


def _load_override_file(path: Path) -> Mapping[str, Any]:
    """Load a flat mapping of dotted overrides from JSON or YAML."""
    import yaml

    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError(
            f"Override file {str(path)!r} must contain a mapping."
        )

    return data


def apply_overrides(
    config: Mapping[str, ...],
    overrides: Iterable[str] | None = None,
) -> dict[str, ...]:
    """Apply ordered dotted overrides or JSON/YAML override files."""
    result = deepcopy(dict(config))

    for override in overrides or ():
        path = Path(override)

        if path.suffix.lower() in {".json", ".yaml", ".yml"}:
            entries = _load_override_file(path).items()
        else:
            try:
                key, raw_value = override.split("=", 1)
            except ValueError as error:
                raise ValueError(
                    "Config overrides must have form "
                    "'path.to.option=value' or specify a "
                    f"JSON/YAML file, but got {override!r}."
                ) from error

            entries = [(key, _parse_override_value(raw_value))]

        for key, value in entries:
            if not isinstance(key, str):
                raise ValueError(
                    f"Config override keys must be strings: {key!r}."
                )

            keys = key.split(".")

            if any(not part for part in keys):
                raise ValueError(
                    f"Invalid config override path {key!r}."
                )

            cursor = result
            for part in keys[:-1]:
                if part not in cursor:
                    cursor[part] = {}
                elif not isinstance(cursor[part], Mapping):
                    raise ValueError(
                        f"Cannot override {key!r}: "
                        f"{part!r} is not a mapping."
                    )
                cursor = cursor[part]

            cursor[keys[-1]] = deepcopy(value)

    return result



def model_aliases() -> tuple[str, ...]:
    """Return bundled model configuration aliases."""

    model_dir = files("arvore").joinpath(
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
        files("arvore")
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



def load_training_config(source):
    """Load a training experiment or reconstruct one from a Box checkpoint.

    Accepted inputs:
      * Experiment config with a 'box' section
      * Checkpoint directory containing config.json
      * Checkpoint manifest JSON
      * Standalone Box.to_config() JSON
      * Equivalent mappings

    Checkpoint-derived configs intentionally contain only 'box':
    fit/save options are not inferred from prior training runs.
    """
    from providence.io import load_json

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / "config.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        source = load_json(str(path))

    if not isinstance(source, Mapping):
        raise TypeError("Training configuration must be a mapping")

    source = deepcopy(dict(source))

    if "box" in source:
        if not isinstance(source["box"], Mapping):
            raise TypeError("Experiment 'box' must be a mapping")
        return source

    # Checkpoint manifest: extract only its construction recipe.
    if "artefacts" in source and "config" in source:
        box = source["config"]
        if not isinstance(box, Mapping):
            raise ValueError("Checkpoint has no Box construction config")
        return {"box": deepcopy(dict(box))}

    # Standalone Box.to_config().
    if all(key in source for key in ("model", "features", "target")):
        return {"box": source}

    raise ValueError(
        "Expected an experiment config, a Box construction config, "
        "or a Arvore checkpoint manifest"
    )
