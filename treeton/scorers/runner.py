"""Scorer lifecycle and shared resource orchestration."""

from dataclasses import replace
from inspect import Parameter, signature

import numpy as np

from .base import Requirements
from .registry import scorer_name


class ScoringRunner:
    """One prediction invocation and its prepared scorer states."""

    def __init__(self, box, scorers):
        self.box = box
        self.scorers = tuple(scorers)
        self.states = ()
        self.needs_trees = False
        self.needs_leaves = False

    def prepare_candidates(self, data):
        """Allow scorers to augment candidate columns."""
        for scorer in self.scorers:
            hook = getattr(
                scorer, "prepare_candidates", None
            )
            if not callable(hook):
                continue

            # Support both prepare_candidates(data) and the
            # new prepare_candidates(data, *, box=...) form.
            parameters = signature(hook).parameters
            accepts_box = (
                "box" in parameters
                or any(
                    p.kind == Parameter.VAR_KEYWORD
                    for p in parameters.values()
                )
            )

            data = (
                hook(data, box=self.box)
                if accepts_box
                else hook(data)
            )

            if data is None:
                raise ValueError(
                    f"{scorer_name(scorer)!r}.prepare_candidates() "
                    "returned None"
                )

        return data

    def validate(self, prepared):
        """Validate shared inference resources and scorer hooks."""
        from ..box.base import _col_names
        from ..models.adapters import get_adapter
        from ..models.ensemble import Ensemble

        columns = set(_col_names(prepared))
        model = self.box.model
        is_ensemble = isinstance(model, Ensemble)

        models = (
            model.members
            if is_ensemble
            else {"model": model}
        )

        allowed_outputs = {
            "prediction",
            "member_predictions",
            "tree_predictions",
            "leaf_indices",
        }
        allowed_capabilities = {
            "ensemble_members",
            "tree_predictions",
            "leaf_indices",
        }

        for scorer in self.scorers:
            name = scorer_name(scorer)
            req = getattr(
                scorer, "requirements", Requirements()
            )

            if not isinstance(req, Requirements):
                raise TypeError(
                    f"Scorer {name!r} must expose Requirements"
                )

            if (
                set(req.outputs) - allowed_outputs
                or set(req.capabilities) - allowed_capabilities
            ):
                raise NotImplementedError(
                    f"Unsupported shared resources for {name!r}"
                )

            missing = set(req.columns) - columns
            if missing:
                raise ValueError(
                    f"Scorer {name!r} requires missing "
                    f"columns: {sorted(missing)}"
                )

            needs_members = (
                "member_predictions" in req.outputs
                or "ensemble_members" in req.capabilities
            )

            if needs_members and (
                not is_ensemble or len(model.members) < 2
            ):
                raise ValueError(
                    f"Scorer {name!r} requires an ensemble "
                    "with at least two members"
                )

            for resource in ("tree_predictions", "leaf_indices"):
                requested = (
                    resource in req.outputs
                    or resource in req.capabilities
                )

                if not requested:
                    continue

                unsupported = [
                    key
                    for key, member in models.items()
                    if resource not in (
                        get_adapter(member).capabilities(member)
                    )
                ]

                if unsupported:
                    raise ValueError(
                        f"Scorer {name!r} requires {resource}; "
                        f"unsupported members: {unsupported}"
                    )

                if resource == "tree_predictions":
                    self.needs_trees = True
                else:
                    self.needs_leaves = True

            hook = getattr(scorer, "validate", None)
            if callable(hook):
                hook(self.box, prepared)

    def prepare(self, *, batch_size):
        """Prepare private reference state once per scorer."""
        states = []

        for scorer in self.scorers:
            hook = getattr(scorer, "prepare", None)
            states.append(
                hook(self.box, batch_size=batch_size)
                if callable(hook)
                else None
            )

        self.states = tuple(states)

    def score(self, context, *, occupied=()):
        """Score a batch without modifying model predictions."""
        if len(self.states) != len(self.scorers):
            raise RuntimeError(
                "ScoringRunner.prepare() has not completed"
            )

        output = {}
        occupied = set(occupied)

        n_samples = len(context.X)
        target_shape = (
            n_samples,
            len(context.targets),
        )

        for scorer, state in zip(
            self.scorers, self.states
        ):
            name = scorer_name(scorer)

            local_context = replace(
                context,
                box=self.box,
                state=state,
            )

            # Deliberately do not mask exceptions raised
            # within the scorer itself.
            values = scorer(local_context)

            try:
                values = np.asarray(
                    values, dtype=float
                )
            except (ValueError, TypeError) as error:
                raise TypeError(
                    f"Scorer {name!r} returned "
                    "nonnumeric values"
                ) from error

            if values.shape == (n_samples,):
                columns = {
                    f"score:{name}": values
                }

            elif values.shape == target_shape:
                columns = {
                    f"score:{name}:{target}": column
                    for target, column in zip(
                        context.targets,
                        values.T,
                    )
                }

            else:
                raise ValueError(
                    f"Scorer {name!r} returned shape "
                    f"{values.shape}; expected "
                    f"({n_samples},) or {target_shape}"
                )

            collisions = set(columns) & (
                occupied | set(output)
            )

            if collisions:
                raise ValueError(
                    "Score columns already exist: "
                    f"{sorted(collisions)}"
                )

            output.update(columns)

        return output
