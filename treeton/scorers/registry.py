"""Registration and resolution of prediction scorers."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .base import Scorer, ScoreContext


SCORER_REGISTRY: dict[str, type[Scorer]] = {}



@dataclass(frozen=True)
class NamedScorer:
    """Delegate to a scorer while overriding its output name."""

    name: str
    scorer: object

    def __getattr__(self, attribute):
        return getattr(self.scorer, attribute)

    def __call__(self, context):
        return self.scorer(context)


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
        raise ValueError(f"Scorer must have a nonempty name, but scorer was {scorer}")

    return name


def resolve_scorers(score):
    """Resolve strings, scorer instances and plain callables."""

    if score is None:
        return ()

    if isinstance(score, Mapping):
        aliases = tuple(score)

        if not all(
            isinstance(name, str) and name
            for name in aliases
        ):
            raise ValueError(
                "Scorer aliases must be nonempty strings"
            )

        specifications = tuple(score.values())

    else:
        aliases = None

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

        if aliases is not None:
            scorer = NamedScorer(
                name=aliases[len(resolved)],
                scorer=scorer,
            )

        resolved.append(scorer)

    names = [scorer_name(s) for s in resolved]

    if len(names) != len(set(names)):
        raise ValueError(
            f"Duplicate scorer names: {names}"
        )

    return tuple(resolved)
