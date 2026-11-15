"""Explicitly unsafe whole-object persistence for arbitrary estimators."""

from .base import register_adapter
from .sklearn import SklearnAdapter


@register_adapter(name="joblib", ext="joblib")
class JoblibAdapter(SklearnAdapter):
    """Explicitly unsafe whole-object persistence."""

    def save(self, model, path, **kwargs):
        import joblib
        joblib.dump(model, path)

    def load(self, path, *, trusted=(), allow_unsafe=False, **kwargs):
        if not allow_unsafe:
            raise ValueError(
                "Joblib checkpoints require allow_unsafe=True"
            )
        import joblib
        return joblib.load(path)
