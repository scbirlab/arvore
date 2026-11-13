"""Preparation, validation and execution of prediction scorers."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from .base import Requirements


class ScoringRunner:
    """Scorer lifecycle for one Box.predict() invocation."""

    def __init__(self, box, scorers: Mapping):
        self.box = box
        self.scorers = dict(scorers)
        self.states = {}
        self.needs_trees = False
        self.needs_leaves = False
        self._prepared = False

    def prepare_candidates(self, data):
        """Allow scorers to prepare auxiliary candidate columns."""
        for name, scorer in self.scorers.items():
            hook = getattr(
                scorer, "prepare_candidates", None
            )

            if not callable(hook):
                continue

            data = hook(data, box=self.box)

            if data is None:
                raise ValueError(
                    f"{name!r}.prepare_candidates() "
                    "returned None"
                )

        return data

    def validate(self, prepared):
        """Check shared resources and run scorer-specific validation."""
        from ..box.base import _col_names
        from ..models.adapters import get_adapter
        from ..models.ensemble import Ensemble

        columns = set(_col_names(prepared))
        model = self.box.model
        is_ensemble = isinstance(model, Ensemble)

        members = (
            model.members
            if is_ensemble
            else {"model": model}
        )

        supported_outputs = {
            "prediction",
            "member_predictions",
            "tree_predictions",
            "leaf_indices",
        }

        supported_capabilities = {
            "ensemble_members",
            "tree_predictions",
            "leaf_indices",
        }

        self.needs_trees = False
        self.needs_leaves = False

        for name, scorer in self.scorers.items():
            req = getattr(
                scorer,
                "requirements",
                Requirements(),
            )

            if not isinstance(req, Requirements):
                raise TypeError(
                    f"Scorer {name!r} must declare Requirements"
                )

            unsupported = (
                set(req.outputs) - supported_outputs
            ) | (
                set(req.capabilities) - supported_capabilities
            )

            if unsupported:
                raise NotImplementedError(
                    f"Scorer {name!r} requests unsupported "
                    f"resources: {sorted(unsupported)}"
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
                not is_ensemble or len(members) < 2
            ):
                raise ValueError(
                    f"Scorer {name!r} requires an ensemble "
                    "with at least two members"
                )

            for capability in (
                "tree_predictions",
                "leaf_indices",
            ):
                requested = (
                    capability in req.outputs
                    or capability in req.capabilities
                )

                if not requested:
                    continue

                unsupported_members = [
                    key
                    for key, member in members.items()
                    if capability not in (
                        get_adapter(member).capabilities(member)
                    )
                ]

                if unsupported_members:
                    raise ValueError(
                        f"Scorer {name!r} requires {capability}; "
                        "unsupported members: "
                        f"{unsupported_members}"
                    )

                if capability == "tree_predictions":
                    self.needs_trees = True

                if capability == "leaf_indices":
                    self.needs_leaves = True

            hook = getattr(scorer, "validate", None)

            if callable(hook):
                hook(self.box, prepared)

    def prepare(self, *, batch_size):
        """Prepare scorer-private state once per prediction call."""
        self.states = {}

        for name, scorer in self.scorers.items():
            hook = getattr(scorer, "prepare", None)

            state = (
                hook(self.box, batch_size=batch_size)
                if callable(hook)
                else None
            )

            self.states[name] = state

        self._prepared = True

    def finalize(self, result):
        """Run post-prediction checks on completed results."""
        for name, scorer in self.scorers.items():
            hook = getattr(scorer, "finalize", None)
            if callable(hook):
                hook(result, name=name, box=self.box)

    def score(self, context, *, occupied=()):
        """Return named score columns for one candidate batch."""
        if not self._prepared:
            raise RuntimeError(
                "ScoringRunner.prepare() must be called first"
            )

        output = {}
        occupied = set(occupied)

        n_samples = len(context.X)
        target_shape = (
            n_samples,
            len(context.targets),
        )

        for name, scorer in self.scorers.items():
            local_context = replace(
                context,
                box=self.box,
                state=self.states[name],
            )

            # A scorer may return its usual numeric array, or a
            # mapping with "" for its primary score and named extras.
            values = scorer(local_context)

            if isinstance(values, Mapping):
                if "" not in values:
                    raise ValueError(
                        f"Scorer {name!r} mapping lacks primary score"
                    )
                outputs = values.items()
            else:
                outputs = (("", values),)

            columns = {}
            for suffix, value in outputs:
                if not isinstance(suffix, str):
                    raise TypeError("Scorer output suffix must be a string")

                column = f"score:{name}"
                if suffix:
                    column += f":{suffix}"

                array = np.asarray(value)

                if array.dtype.kind in "US":
                    if not suffix or array.shape != (n_samples,):
                        raise ValueError(
                            f"Scorer {name!r} returned invalid string shape"
                        )
                    columns[column] = array.tolist()
                    continue

                try:
                    array = np.asarray(value, dtype=float)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"Scorer {name!r} returned nonnumeric values"
                    ) from error

                if array.shape == (n_samples,):
                    columns[column] = array
                elif array.shape == target_shape and not suffix:
                    columns.update({
                        f"{column}:{target}": array[:, i]
                        for i, target in enumerate(context.targets)
                    })
                else:
                    raise ValueError(
                        f"Scorer {name!r} returned shape "
                        f"{array.shape}; expected ({n_samples},)"
                        f" or {target_shape}"
                    )

            collisions = set(columns) & (
                occupied | set(output)
            )

            if collisions:
                raise ValueError(
                    "Score output columns already exist: "
                    f"{sorted(collisions)}"
                )

            output.update(columns)

        return output
