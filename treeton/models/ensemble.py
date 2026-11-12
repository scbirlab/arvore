"""Composable regression ensembles."""

from collections.abc import Callable, Mapping, Iterable
from copy import deepcopy
from math import isfinite
from pathlib import Path
import os

from ..typing import Estimator

DEFAULT_SEED: int = 42

class Ensemble:
    """A named collection of independently fitted estimators.

    All members share the same feature matrix and target.
    Predictions are combined using a weighted arithmetic mean.
    """

    SCHEMA_VERSION: int = 1

    def __init__(
        self,
        members: Mapping[str, ...] | Iterable[Estimator] | Estimator,
        *,
        weights=None
    ):
        if isinstance(members, Estimator):
            members = (members,)
        if isinstance(members, Iterable) and not isinstance(members, (str, Mapping)):
            members = tuple(members)
            if len(members) == 1:
                members = {"model": members[0]}
            else:
                members = {f"model:{i}": m for i, m in enumerate(members)}
        if len(members) == 0:
            raise ValueError(f"members must be nonempty: {members}")
        
        self.members = dict(members)

        if not all(
            isinstance(name, str) and name
            for name in self.members
        ):
            raise ValueError(f"Member names must be nonempty strings, but were: {', '.join(map(str, self.members))}")

        if len({id(m) for m in self.members.values()}) != len(
            self.members
        ):
            raise ValueError(
                "Members must be distinct estimator instances"
            )

        for name, member in self.members.items():
            if isinstance(member, Ensemble):
                raise ValueError("Nested ensembles are not supported")
            if not all(
                callable(getattr(member, method, None))
                for method in ("fit", "predict")
            ):
                raise TypeError(
                    f"Member {name} must implement fit and predict"
                )

        if weights is None:
            weights = [1.] * len(self.members)

        self.weights = tuple(float(w) for w in weights)

        if len(self.weights) != len(self.members):
            raise ValueError("Expected one weight per member")
        if any(
            not isfinite(w) or w <= 0
            for w in self.weights
        ):
            raise ValueError(
                "Weights must be positive and finite"
            )
        self._config = None

    @classmethod
    def repeat(
        cls,
        base,
        n: int,
        *,
        seed: int | None = DEFAULT_SEED,
        weights=None,
    ) -> "Ensemble":
        """Clone an estimator with distinct deterministic seeds."""
        from sklearn.base import clone

        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, but was {n}")

        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an integer or None, but was {seed}")

        members = []

        for i in range(n):
            member = clone(base)

            if seed is not None:
                params = member.get_params(deep=False)

                if "random_state" in params:
                    key = "random_state"
                elif "random_seed" in params:
                    key = "random_seed"
                else:
                    raise ValueError(
                        "Estimator has no random_state or "
                        "random_seed parameter; pass seed=None "
                        "or use explicit members"
                    )

                member.set_params(**{key: int(seed) + i})

            members.append(member)

        return cls(members, weights=weights)

    def __len__(self):
        return len(self.members)

    @property
    def has_config(self) -> bool:
        return self._config is not None

    def _attach_config(self, config):
        """Attach a declarative specification without replacing an existing one."""
        from ..config import parse_model_config

        parsed = parse_model_config(config)

        if self._config is not None and self._config != parsed:
            raise ValueError(
                "Model configuration conflicts with the "
                "ensemble's existing specification:\n"
                f"Incumbent: {self._config}\n"
                f"Parsed: {parsed}\n"
            )

        self._config = parsed

    def to_config(self, path=None):
        """Return the original declarative model specification."""
        if self._config is None:
            raise ValueError(
                "This ensemble was constructed from Python estimators "
                "and has no declarative configuration."
            )

        config = self._config.to_dict()

        if path is not None:
            from providence.io import save_json
            save_json(config, path)

        return config

    @classmethod
    def from_config(cls, config):
        """Construct an ensemble, retaining its model specification."""
        from ..config import (
            EstimatorConfig,
            parse_model_config,
            instantiate_object,
        )
        from providence.io import load_json

        if isinstance(config, (str, os.PathLike)):
            config = load_json(str(config))

        parsed = parse_model_config(config)

        if isinstance(parsed, EstimatorConfig):
            ensemble = cls(
                instantiate_object(parsed.to_dict())
            )

        elif parsed.members is not None:
            ensemble = cls(
                {
                    name: instantiate_object(spec.to_dict())
                    for name, spec in parsed.members.items()
                },
                weights=parsed.weights,
            )

        else:
            ensemble = cls.repeat(
                instantiate_object(parsed.base.to_dict()),
                n=parsed.n_members,
                seed=parsed.seed,
                weights=parsed.weights,
            )

        ensemble._config = parsed
        return ensemble


    def save(
        self, 
        path: str | os.PathLike, 
        *, 
        allow_unsafe: bool = False
    ):
        from providence.io import save_json
        from ..config import EnsembleManifest
        from .adapters import save_model
        path = Path(path)
        entries = []

        for index, (name, member) in enumerate(self.members.items()):
            directory = path.parent / "members" / f"{index:04d}"

            artifact = save_model(
                member,
                directory,
                allow_unsafe=allow_unsafe,
            )

            entries.append({
                "name": name,
                "metadata": artifact,
            })

        manifest = EnsembleManifest.from_dict({
            "schema_version": EnsembleManifest._version,
            "config": self.to_config() if self.has_config else None,
            "weights": list(self.weights),
            "members": entries,
        })

        save_json(manifest.to_dict(), path)
        return None


    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        trusted: Iterable[str] = (),
        allow_unsafe: bool = False,
    ):
        from providence.io import load_json
        from ..config import EnsembleManifest
        from .adapters import load_model

        path = Path(path)
        manifest = EnsembleManifest.from_dict(load_json(path))

        members = {}

        for index, (name, artifact) in enumerate(manifest.members):
            directory = path.parent / "members" / f"{index:04d}"

            members[name] = load_model(
                directory,
                artifact.to_dict(),
                allow_unsafe=allow_unsafe,
                trusted_types=trusted,
            )

        ensemble = cls(members, weights=manifest.weights)

        if manifest.config is not None:
            ensemble._config = manifest.config

        return ensemble

    def fit(
        self,
        X,
        y,
        *,
        validation=None,
        early_stopping_rounds=None,
        **fit_kwargs,
    ):
        from .adapters import get_adapter

        get_adapter(self).fit(
            self,
            X,
            y,
            validation=validation,
            fit_kwargs=fit_kwargs,
            early_stopping_rounds=early_stopping_rounds,
        )
        return self

    def predict_members(
        self, 
        X,
        **kwargs
    ):
        import numpy as np
        from .adapters import get_adapter

        predictions = []

        for name, member in self.members.items():
            adapter = get_adapter(member)
            prediction = np.asarray(
                adapter.predict(member, X, **kwargs),
                dtype=float,
            )

            if prediction.ndim == 1:
                prediction = prediction[:, None]

            if prediction.ndim != 2 or prediction.shape[0] != len(X):
                raise ValueError(
                    f"Invalid prediction shape for {name!r}: "
                    f"{prediction.shape}"
                )

            if predictions and prediction.shape != predictions[0].shape:
                raise ValueError(
                    f"Incompatible prediction shape for {name!r}"
                )
            predictions.append(prediction)
        return np.stack(predictions, axis=1)

    def predict( 
        self, 
        X, 
        *, 
        disaggregate: bool = False, 
        **kwargs
    ):
        import numpy as np

        members = self.predict_members(X, **kwargs)

        if not disaggregate:
            prediction = np.average(
                members,
                axis=1,
                weights=self.weights,
            )
        else:
            prediction = members

        if not disaggregate and prediction.shape[1] == 1:
            return prediction[:, 0]

        return prediction

