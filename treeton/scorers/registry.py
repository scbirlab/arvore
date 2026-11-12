"""Registration and resolution of prediction scorers."""

from collections.abc import Callable, Iterable

from .base import Scorer, ScoreContext


SCORER_REGISTRY: dict[str, type[Scorer]] = {}


def register_scorer(cls):
    """Register a named scorer class."""

    name = getattr(cls, "name", None)

    if not isinstance(name, str) or not name:
        raise ValueError("Registered scorers require a name")

    if name in SCORER_REGISTRY:
        raise ValueError(f"Scorer {name!r} already registered")

    SCORER_REGISTRY[name] = cls
    return cls


def scorer_name(scorer) -> str:
    """Resolve a stable name for a scorer or callable."""

    name = getattr(scorer, "name", None)

    if name is None:
        name = getattr(
            scorer,
            "__name__",
            type(scorer).__name__,
        )

    if not isinstance(name, str) or not name:
        raise ValueError("Scorer must have a nonempty name")

    return name


def resolve_scorers(score):
    """Resolve strings, scorer instances and plain callables."""

    if score is None:
        return ()

    if isinstance(score, str) or callable(score):
        specifications = (score,)
    else:
        specifications = tuple(score)

    resolved = []

    for specification in specifications:
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
                "Scorers must be names or callables, "
                f"received {type(specification).__name__}"
            )

        resolved.append(scorer)

    names = [scorer_name(s) for s in resolved]

    if len(names) != len(set(names)):
        raise ValueError(
            f"Duplicate scorer names: {names}"
        )

    return tuple(resolved)
