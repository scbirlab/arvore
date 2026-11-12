from typing import ClassVar
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, asdict
import os
from math import isfinite


def _freeze(value):
    """Recursively freeze a JSON-compatible value."""
    if isinstance(value, Mapping):
        return FrozenDict({
            k: _freeze(v) 
            for k, v in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not isfinite(value):
            raise ValueError(f"Non-finite config value: {valurf}")
        return value
    raise TypeError(
        f"Not JSON-compatible: {type(value).__name__}"
    )


def _thaw(value):
    """Return a fresh, mutable JSON-compatible value."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, but was {value}")


def _weights(value, n):
    if value is None:
        return None

    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError("Expected one weight per member")

    if any(
        type(w) not in (int, float)
        or not isfinite(w)
        or w <= 0
        for w in value
    ):
        raise ValueError(f"Weights must be positive finite numbers, but were {value}")

    return tuple(value)


class FrozenDict(dict):
    def __setitem__(self, key, value):
        raise TypeError("Cannot alter a FrozenDict")


@dataclass(frozen=True)
class EstimatorConfig:
    class_path: str
    init_args: Mapping[str, ...] = field(default_factory=dict)
    _raw: Mapping[str, ...] = field(repr=False, compare=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise TypeError("Estimator config must be a mapping")

        extra_fields = set(data) - set(n.name for n in fields(cls))
        if extra_fields:
            raise ValueError(f"Extra config fields: {extra_fields}")

        path = data.get("class_path")
        args = data.get("init_args", {})

        if not isinstance(path, str) or not path.strip():
            raise ValueError("class_path must be nonempty")

        if not isinstance(args, Mapping):
            raise TypeError("init_args must be a mapping")

        raw = _freeze(data)
        return cls(path, _freeze(args), raw)

    def to_dict(self, raw: bool = False):
        if raw:
            return _thaw(self._raw)
        else:
            return _thaw(asdict(self))


@dataclass(frozen=True)
class EnsembleConfig:
    base: EstimatorConfig | None
    members: Mapping[str, EstimatorConfig] | None
    n_members: int
    seed: int | None
    weights: tuple[int | float, ...] | None
    schema_version: int | None = None
    _raw: Mapping[str, ...] = field(repr=False, compare=False, default_factory=dict)
    _version: ClassVar = 1

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise TypeError("Ensemble config must be a mapping")

        allowed = {
            "kind", "schema_version", "base",
            "members", "n_members", "seed", "weights",
        }
        if set(data) - allowed:
            raise ValueError("Unknown ensemble config fields")

        if data.get("kind") != "ensemble":
            raise ValueError("Expected kind='ensemble'")

        schema = data.get("schema_version", cls._version)
        if type(schema) is not int or schema != cls._version:
            raise ValueError(
                f"Unsupported ensemble schema {schema} "
                f"(package uses {cls._version}"
            )

        has_base = "base" in data
        has_members = "members" in data

        if has_base and has_members:
            raise ValueError("Use `base` or `members`")

        if has_members:
            if "n_members" in data or "seed" in data:
                raise ValueError(
                    "`n_members` and `seed` require `base`"
                )

            specs = data["members"]
            if not isinstance(specs, Mapping) or not specs:
                raise ValueError("`members` must be nonempty")

            if any(
                not isinstance(k, str) or not k
                for k in specs
            ):
                raise ValueError("Invalid member name")

            members = FrozenDict({
                name: EstimatorConfig.from_dict(spec)
                for name, spec in specs.items()
            })
            base = None
            count = len(members)
            seed = None

        else:
            base = EstimatorConfig.from_dict(data["base"])
            members = None
            count = data.get("n_members", 1)
            _positive_int(count, "n_members")

            seed = data.get("seed", 42)
            if seed is not None and type(seed) is not int:
                raise ValueError(
                    f"`seed` must be an integer or None, but was {seed}"
                )

        weights = _weights(data.get("weights"), count)

        return cls(
            schema_version=schema,
            base=base,
            members=members,
            n_members=count,
            seed=seed,
            weights=weights,
            _raw=_freeze(data),
        )

    def to_dict(self):
        return _thaw(self._raw)


def parse_model_config(data):
    """Parse either a single estimator or ensemble specification."""
    if not isinstance(data, Mapping):
        raise TypeError("Model config must be a mapping")

    if data.get("kind") == "ensemble":
        return EnsembleConfig.from_dict(data)

    if "kind" in data:
        raise ValueError(f"Unknown model kind: {data['kind']}")

    return EstimatorConfig.from_dict(data)



@dataclass(frozen=True)
class ModelArtifact:
    adapter: str
    filename: str | None
    class_path: str
    version: str
    sha256: str | None
    trainable: bool

    def __post_init__(self):
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError("Invalid adapter identifier")

        if not isinstance(self.class_path, str) or not self.class_path:
            raise ValueError("Invalid model class path")

        if (self.filename is None) != (self.sha256 is None):
            raise ValueError(
                "filename and sha256 must both be set or None"
            )

        if self.filename is not None:
            if (
                not self.filename
                or self.filename in (".", "..")
                or "/" in self.filename
                or "\\" in self.filename
            ):
                raise ValueError("Invalid artifact filename")

            if (
                len(self.sha256) != 64
                or any(
                    c not in "0123456789abcdef"
                    for c in self.sha256
                )
            ):
                raise ValueError("Invalid SHA-256 checksum")

    @classmethod
    def from_dict(cls, data: Mapping):
        if not isinstance(data, Mapping):
            raise TypeError("Artifact must be a mapping")

        expected = set(n.name for n in fields(cls))
        data_fields = set(data)
        if data_fields != expected:
            raise ValueError(
                f"Invalid model artifact fields: {data_fields}\n"
                f"Expected: {expected}"
            )

        return cls(**dict(data))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EnsembleManifest:
    schema_version: int
    config: EstimatorConfig | EnsembleConfig | None
    weights: tuple[float, ...]
    members: tuple[tuple[str, ModelArtifact], ...]
    _version: ClassVar = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != self._version:
            raise ValueError(
                f"Unsupported ensemble manifest schema version {self.schema_version}; "
                f"package uses {self._version}"
            )

        names = [name for name, _ in self.members]
        if (
            any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Invalid or duplicate member names")

        _weights(self.weights, len(self.members))

        if any(
            artifact.adapter == "ensemble"
            for _, artifact in self.members
        ):
            raise ValueError("Nested ensemble artifacts unsupported")

    @classmethod
    def from_dict(cls, data: Mapping):
        if not isinstance(data, Mapping):
            raise TypeError("Manifest must be a mapping")

        expected = set(n.name for n in fields(cls))
        data_fields = set(data)
        if data_fields != expected:
            raise ValueError(
                f"Invalid ensemble manifest fields: {data_fields}\n"
                f"Expected: {expected}"
            )

        entries = data["members"]
        if not isinstance(entries, (list, tuple)):
            raise ValueError("Invalid ensemble members")

        members = []
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {
                "name", 
                "metadata",
            }:
                raise ValueError(f"Invalid member entry: {entry}")

            members.append((
                entry["name"],
                ModelArtifact.from_dict(entry["metadata"]),
            ))

        config = data.get("config")
        if config is not None:
            config = parse_model_config(config)

        weights = _weights(data["weights"], len(members))
        if weights is None:
            raise ValueError("Manifest requires weights")

        return cls(
            schema_version=data["schema_version"],
            config=config,
            weights=tuple(float(w) for w in weights),
            members=tuple(members),
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "config": (
                self.config.to_dict()
                if self.config is not None else None
            ),
            "weights": list(self.weights),
            "members": [
                {
                    "name": name,
                    "metadata": artifact.to_dict(),
                }
                for name, artifact in self.members
            ],
        }

