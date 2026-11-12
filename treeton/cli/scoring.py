"""Resolve CLI scorer names and declarative configurations."""

from collections.abc import Mapping

from ..config.instantiate import instantiate_object
from ..scorers import resolve_scorers


def resolve_cli_scores(config=None, names=None):
    """Resolve configured and CLI-selected scorers.

    Config accepts:
      - a registered scorer name
      - a list of registered names
      - a mapping of output names to registered names or
        class_path/init_args specifications

    Repeated --score options add registered scorer names.
    """
    if config is None:
        specifications = {}

    elif isinstance(config, str):
        specifications = {config: config}

    elif isinstance(config, list):
        if not all(isinstance(s, str) for s in config):
            raise TypeError(
                "List-form scores must be registered names"
            )
        if len(config) != len(set(config)):
            raise ValueError("Duplicate configured scorer names")
        specifications = {name: name for name in config}

    elif isinstance(config, Mapping):
        specifications = dict(config)

    else:
        raise TypeError(
            "score must be a name, list of names, or mapping"
        )

    for name in names or ():
        if not isinstance(name, str) or not name:
            raise ValueError("--score requires a nonempty name")

        if name in specifications:
            raise ValueError(
                f"Duplicate scorer output name: {name!r}"
            )

        specifications[name] = name

    resolved = {}

    for output_name, specification in specifications.items():
        if isinstance(specification, str):
            resolved[output_name] = specification

        elif isinstance(specification, Mapping):
            allowed = {"class_path", "init_args"}

            if (
                "class_path" not in specification
                or set(specification) - allowed
            ):
                raise ValueError(
                    f"Invalid scorer configuration for "
                    f"{output_name!r}; expected class_path "
                    "and optional init_args"
                )

            resolved[output_name] = instantiate_object(
                specification
            )

        else:
            raise TypeError(
                f"Invalid scorer specification for "
                f"{output_name!r}: {specification!r}"
            )

    # Validate registration, aliases and callable instances
    # before loading candidate data.
    return resolve_scorers(resolved) if resolved else None
