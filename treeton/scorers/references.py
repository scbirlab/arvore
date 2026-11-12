"""Per-Box, thread-safe prepared reference resources.

Resource specifications describe *construction*, not scorer aliases. The Box's
fitted model and preprocessing define the scope of every cache instance.
Concurrent read-only predictions are supported; fitting during prediction is not.
"""

from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
import json


def canonical_spec(spec):
    """Validate and copy a canonical, JSON-native resource specification."""
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise ValueError("Reference spec requires a 'type' string")
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    result = json.loads(encoded)
    if result != spec:
        raise ValueError("Reference spec must use JSON-native values")
    return result


def reference_key(spec):
    """Stable SHA-256 identity for a resource's complete construction spec."""
    spec = canonical_spec(spec)
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class PreparedReference:
    spec: dict
    value: object


class ReferenceCache:
    """Store complete references; builders run once under an instance lock.

    Resource builders must return fully prepared, immutable-in-use objects.
    Failed builds do not insert entries. Call clear() after external mutation.
    """

    def __init__(self):
        self._lock = RLock()
        self._entries = {}

    def get_or_build(self, spec, builder):
        spec = canonical_spec(spec)
        key = reference_key(spec)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                if entry.spec != spec:
                    raise ValueError("Reference digest collision")
                return entry.value
            value = builder()
            self._entries[key] = PreparedReference(spec, value)
            return value

    def clear(self):
        with self._lock:
            self._entries.clear()

    def snapshot(self):
        """Atomically capture a consistent set of prepared objects."""
        with self._lock:
            return dict(self._entries)

    def restore(self, entries):
        """Install already-validated resources from a checkpoint."""
        with self._lock:
            self._entries = dict(entries)
