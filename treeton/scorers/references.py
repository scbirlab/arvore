
"""Per-Box cache of prepared reference resources."""

from collections.abc import Callable, Mapping
from hashlib import sha256
from threading import RLock
import json


def _canonical_json(spec):
    return json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_spec(spec):
    """Return a validated JSON-native construction specification."""
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise ValueError("Reference spec requires a 'type' string")

    result = json.loads(_canonical_json(spec))
    if result != spec:
        raise ValueError("Reference spec must use JSON-native values")
    return result


def reference_key(spec):
    """Stable identifier independent of Python object identity."""
    spec = canonical_spec(spec)
    return sha256(_canonical_json(spec).encode()).hexdigest()


class ReferenceCache:
    """Prepare each resource once, scoped to a fitted Box.

    Builders run under a per-instance lock.
    """

    def __init__(self):
        self._lock = RLock()
        self._entries = {}

    def get_or_build(
        self,
        spec: Mapping,
        builder: Callable,
    ):
        spec = canonical_spec(dict(spec))
        key = reference_key(spec)

        with self._lock:
            if key not in self._entries:
                self._entries[key] = (spec, builder())
            return self._entries[key][1]

    def clear(self):
        with self._lock:
            self._entries.clear()

    def snapshot(self):
        """Return a consistent snapshot of prepared entries."""
        with self._lock:
            return dict(self._entries)

    def restore(self, entries):
        """Install resources validated by the checkpoint loader."""
        with self._lock:
            self._entries = dict(entries)
