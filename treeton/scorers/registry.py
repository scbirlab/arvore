"""Registration and resolution of prediction scorers."""

from collections.abc import Mapping

SCORER_REGISTRY = {}


def register_scorer(cls):
    """Register a named scorer class."""
    name = getattr(cls, "name", None)

    if not isinstance(name, str) or not name:
        raise ValueError(
            "Registered scorers require a nonempty name"
        )

    if name in SCORER_REGISTRY:
        raise ValueError(
            f"Scorer {name!r} already registered"
        )

    SCORER_REGISTRY[name] = cls
    return cls


def scorer_name(scorer):
    """Return a default name for a scorer or callable."""
    name = getattr(scorer, "name", None)

    if name is None:
        name = getattr(
            scorer,
            "__name__",
            type(scorer).__name__,
        )

    if not isinstance(name, str) or not name:
        raise ValueError(
            f"Scorer has no valid name: {scorer!r}"
        )

    return name


def resolve_scorers(score):
    """Resolve scorer specifications to {output_name: callable}.

    Supported inputs:
      None
      "registered_name"
      scorer_instance
      callable
      iterable of the above
      mapping of output names to the above

    A mapping explicitly overrides scorer names.
    """
    if score is None:
        return {}

    if isinstance(score, Mapping):
        specifications = score.items()

    elif isinstance(score, str) or callable(score):
        specifications = ((None, score),)

    else:
        try:
            specifications = (
                (None, specification)
                for specification in score
            )
        except TypeError as error:
            raise TypeError(
                "score must be a scorer, iterable, "
                "mapping, or None"
            ) from error

    resolved = {}

    for alias, specification in specifications:
        if isinstance(specification, str):
            try:
                scorer = SCORER_REGISTRY[specification]()
            except KeyError as error:
                raise ValueError(
                    f"Unknown scorer {specification!r}. "
                    f"Available: {sorted(SCORER_REGISTRY)}"
                ) from error

        elif isinstance(specification, type):
            raise TypeError(
                "Pass a scorer instance, not a class"
            )

        elif callable(specification):
            scorer = specification

        else:
            raise TypeError(
                "Scorer specifications must be "
                "registered names or callables"
            )

        name = (
            scorer_name(scorer)
            if alias is None
            else alias
        )

        if not isinstance(name, str) or not name:
            raise ValueError(
                "Scorer output names must be nonempty strings"
            )

        if name in resolved:
            raise ValueError(
                f"Duplicate scorer output name: {name!r}"
            )

        resolved[name] = scorer

    return resolved
