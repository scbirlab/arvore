"""Composable regression ensembles."""

from collections.abc import Callable, Mapping, Iterable
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

    def to_config(
        self,
        path: Mapping[str, ...] | str | os.PathLike | None
    ):
        """Instantiate a single estimator or declarative ensemble."""
        from ..config import instantiate_object

        config = {
            "kind": "ensemble", 
            "schema_version": self.SCHEMA_VERSION,
            "weights": list(self.weights),
            "members": {},
        }

        for name, member in self.members.items():
            config["members"][name] = f"{member.__module__}.{member.__qualname__}"
        if path is not None:
            from providence.io import save_json
            save_json(config, path)
        return config

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, ...] | str | os.PathLike
    ):
        """Instantiate a single estimator or declarative ensemble."""
        from ..config import instantiate_object

        if isinstance(config, str):
            from providence.io import load_json
            path = Path(config)
            if not path.exists():
                raise FileNotFoundError(f"Config file {path} not found")
            config = load_json(config)
        if not isinstance(config, Mapping):
            raise ValueError("Config must be a dict or mapping")

        schema = config.get("schema_version")
        if schema != cls.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported ensemble checkpoint schema: "
                f"version is {schema} but package is {cls.SCHEMA_VERSION}"
            )
        kind = config.get("kind")

        if kind is None:
            return cls(instantiate_object(config))

        if kind != "ensemble":
            raise ValueError(f"Unknown model kind: {kind}")

        if "members" in config:
            if "base" in config or "n_members" in config:
                raise ValueError(
                    "Use either members or base/n_members"
                )

            return cls({
                name: instantiate_object(spec)
                for name, spec in config["members"].items()
            }, weights=config.get("weights"))
        elif "base" in config:
            models = instantiate_object(config["base"])
            return cls.repeat(
                models,
                n=config.get("n_members", 1),
                seed=config.get("seed", DEFAULT_SEED),
                weights=config.get("weights"),
            )
        else:
            raise ValueError(
                "Ensemble config requires base or members"
            )

    def save(
        self, 
        path: str | os.PathLike, 
        *, 
        allow_unsafe: bool = False
    ):
        from providence.io import save_json
        from .codecs import save_model
        path = Path(path)
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "weights": list(self.weights),
            "members": [],
        }

        for index, (name, member) in enumerate(self.members.items()):
            directory = (
                path.parent / "members" / f"{index:04d}"
            )
            # Use the existing codec dispatcher, including
            # backend-native formats and safety controls.
            metadata = save_model(
                member,
                directory,
                allow_unsafe=allow_unsafe,
            )
            manifest["members"].append({
                "name": name,
                "metadata": metadata,
            })
        save_json(manifest, path)
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
        from .codecs import load_model

        path = Path(path)
        manifest = load_json(path)

        schema = manifest.get("schema_version")
        if schema != cls.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported ensemble checkpoint schema: "
                f"version is {schema} but package is {cls.SCHEMA_VERSION}"
            )

        entries = manifest.get("members")
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError("Invalid ensemble member manifest")

        members = {}

        for index, entry in enumerate(entries):
            name = entry["name"]
            metadata = entry["metadata"]

            if not isinstance(name, str) or not name or name in members:
                raise ValueError("Invalid or duplicate member name")

            if metadata["codec"] == "ensemble":
                raise ValueError(
                    "Nested ensemble checkpoints are unsupported"
                )
            directory = (
                path.parent / "members" / f"{index:04d}"
            )
            members[name] = load_model(
                directory,
                metadata,
                allow_unsafe=allow_unsafe,
                trusted_types=trusted,
            )

        return cls(
            members,
            weights=manifest["weights"],
        )

    def fit(self, X, y, validation=None, **kwargs):
        """Fit through member-specific adapters."""
        from .adapters import get_adapter

        for member in self.members.values():
            adapter = get_adapter(member)
            adapter.fit(
                member,
                X,
                y,
                validation=validation,
                **kwargs,
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

        if prediction.shape[1] == 1:
            return prediction[:, 0]

        return prediction

