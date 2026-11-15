"""Adapter for arvore ensembles, persisted as a manifest of member checkpoints."""

from .base import register_adapter
from .sklearn import SklearnAdapter


@register_adapter(name="ensemble", ext="json", package="arvore")
class EnsembleAdapter(SklearnAdapter):

    def adapter_name(self, estimator, *, allow_unsafe=False):
        return self.name

    def save(self, model, path, *, allow_unsafe=False, **kwargs):
        model.save(path, allow_unsafe=allow_unsafe)

    def load(
        self, path, *, trusted=(), allow_unsafe=False, **kwargs
    ):
        from ..ensemble import Ensemble

        return Ensemble.load(
            path,
            trusted=trusted,
            allow_unsafe=allow_unsafe,
        )

    def fit(
        self,
        estimator,
        X,
        y,
        *,
        validation=None,
        fit_kwargs=None,
        early_stopping_rounds=None,
        loss=None,
        censor=None,
    ):
        from .dispatch import get_adapter

        for member in estimator.members.values():
            adapter = get_adapter(member)
            adapter.fit(
                member,
                X,
                y,
                validation=validation,
                fit_kwargs=fit_kwargs,
                early_stopping_rounds=early_stopping_rounds,
                loss=loss,
                censor=censor,
            )
