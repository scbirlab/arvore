"""Resolve registered scorers and lightweight CLI configurations."""

import ast
import math
import re
from collections.abc import Mapping

from ..config.instantiate import instantiate_object
from ..scorers import SCORER_REGISTRY, resolve_scorers


# Convenience names for CLI/config expressions only.
# Registered scorer names remain unchanged.
CLI_SCORER_ALIASES = {
    "tanimoto": "tanimoto_distance",
    "feature": "feature_distance",
}


def _literal(node):
    """Parse literal arguments without evaluating Python expressions."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError) as error:
        raise ValueError(
            "Scorer arguments must be Python literals"
        ) from error

    def validate(item):
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Non-finite scorer arguments are unsupported")
            return
        if isinstance(item, (list, tuple)):
            for element in item:
                validate(element)
            return
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Scorer argument dictionary keys must be strings")
            for element in item.values():
                validate(element)
            return
        raise ValueError(
            f"Unsupported scorer argument type: {type(item).__name__}"
        )

    validate(value)
    return value


def parse_scorer(expression):
    """Resolve 'name' or 'name(key=value,...)' to a scorer instance.

    Only registered scorer classes can be instantiated.
    Positional arguments and arbitrary Python expressions are rejected.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Scorer expression must be a nonempty string")

    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as error:
        raise ValueError(
            f"Invalid scorer expression: {expression!r}"
        ) from error

    if isinstance(tree, ast.Name):
        name = tree.id
        kwargs = {}
    elif isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name):
        name = tree.func.id
        if tree.args:
            raise ValueError("Scorer expressions accept keyword arguments only")
        if any(keyword.arg is None for keyword in tree.keywords):
            raise ValueError("**kwargs expansion is unsupported")
        kwargs = {
            keyword.arg: _literal(keyword.value)
            for keyword in tree.keywords
        }
    else:
        raise ValueError(
            "Expected a registered scorer name or name(key=value,...)"
        )

    registered = CLI_SCORER_ALIASES.get(name, name)

    if registered not in SCORER_REGISTRY:
        available = sorted(
            set(SCORER_REGISTRY) | set(CLI_SCORER_ALIASES)
        )
        raise ValueError(
            f"Unknown scorer {name!r}. Available: {available}"
        )

    scorer_class = SCORER_REGISTRY[registered]

    try:
        return scorer_class(**kwargs)
    except TypeError as error:
        raise ValueError(
            f"Invalid arguments for scorer {name!r}: {error}"
        ) from error


def _split_alias(expression):
    """Split optional 'output_name=scorer(...)' syntax."""
    expression = expression.strip()
    match = re.fullmatch(
        r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+)",
        expression,
    )
    if match:
        return match.group(1), match.group(2).strip()

    # For an unaliased expression, the invoked name becomes
    # the output name (including convenient short aliases).
    match = re.match(r"^([A-Za-z_][A-Za-z_0-9]*)", expression)
    if match is None:
        raise ValueError(f"Invalid scorer specification: {expression!r}")

    return match.group(1), expression


def _add(specifications, output_name, specification):
    if not isinstance(output_name, str) or not output_name:
        raise ValueError("Scorer output name must be nonempty")
    if output_name in specifications:
        raise ValueError(
            f"Duplicate scorer output name: {output_name!r}"
        )
    specifications[output_name] = specification


def resolve_cli_scores(config=None, names=None):
    """Merge config scorers and --score arguments.

    Config supports registered names, name(kwargs) expressions,
    lists of expressions, and mappings with explicitly named outputs.

    A mapping value may also be a class_path/init_args configuration.

    --score adds expressions; an optional output_name= prefix
    permits multiple configurations of the same registered scorer.
    """
    specifications = {}

    if config is None:
        pass
    elif isinstance(config, str):
        name, expression = _split_alias(config)
        _add(specifications, name, parse_scorer(expression))
    elif isinstance(config, list):
        for item in config:
            if not isinstance(item, str):
                raise TypeError("List-form scorers must be strings")
            name, expression = _split_alias(item)
            _add(specifications, name, parse_scorer(expression))
    elif isinstance(config, Mapping):
        for name, specification in config.items():
            if isinstance(specification, str):
                resolved = parse_scorer(specification)
            elif isinstance(specification, Mapping):
                allowed = {"class_path", "init_args"}
                if (
                    "class_path" not in specification
                    or set(specification) - allowed
                ):
                    raise ValueError(
                        f"Invalid scorer configuration for {name!r}; "
                        "expected class_path and optional init_args"
                    )
                resolved = instantiate_object(specification)
            else:
                raise TypeError(
                    f"Invalid scorer specification for {name!r}"
                )
            _add(specifications, name, resolved)
    else:
        raise TypeError(
            "score must be a name, list of names, or mapping"
        )

    for item in names or ():
        if not isinstance(item, str):
            raise TypeError("--score arguments must be strings")
        name, expression = _split_alias(item)
        _add(specifications, name, parse_scorer(expression))

    return resolve_scorers(specifications) if specifications else None
