from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType


# ---- Typed model configurations --------------------------------

def _freeze(value):
    """Recursively freeze a JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType({
            k: _freeze(v) for k, v in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not isfinite(value):
            raise ValueError("Non-finite config value")
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
        raise ValueError(f"{name} must be a positive integer")


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
        raise ValueError("Weights must be positive finite numbers")

    return tuple(value)


@dataclass(frozen=True)
class EstimatorConfig:
    class_path: str
    init_args: Mapping[str, Any]
    _raw: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise TypeError("Estimator config must be a mapping")

        if set(data) - {"class_path", "init_args"}:
            raise ValueError("Unknown estimator config fields")

        path = data.get("class_path")
        args = data.get("init_args", {})

        if not isinstance(path, str) or not path.strip():
            raise ValueError("class_path must be nonempty")

        if not isinstance(args, Mapping):
            raise TypeError("init_args must be a mapping")

        raw = _freeze(data)
        return cls(path, _freeze(args), raw)

    def to_dict(self):
        return _thaw(self._raw)


@dataclass(frozen=True)
class EnsembleConfig:
    schema_version: int
    base: EstimatorConfig | None
    members: Mapping[str, EstimatorConfig] | None
    n_members: int
    seed: int | None
    weights: tuple[int | float, ...] | None
    _raw: Mapping[str, Any] = field(repr=False, compare=False)

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

        schema = data.get("schema_version", 1)
        if type(schema) is not int or schema != 1:
            raise ValueError("Unsupported ensemble schema")

        has_base = "base" in data
        has_members = "members" in data

        if has_base == has_members:
            raise ValueError("Use exactly one of base or members")

        if has_members:
            if "n_members" in data or "seed" in data:
                raise ValueError(
                    "n_members and seed require base"
                )

            specs = data["members"]
            if not isinstance(specs, Mapping) or not specs:
                raise ValueError("members must be nonempty")

            if any(
                not isinstance(k, str) or not k
                for k in specs
            ):
                raise ValueError("Invalid member name")

            members = MappingProxyType({
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
                raise ValueError("seed must be an integer or None")

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
