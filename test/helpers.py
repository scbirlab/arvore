"""Shared test construction helpers."""

from types import SimpleNamespace

from arvore import Box, Ensemble


def bare_box(model, features, target=("y",), *, reference=None, candidates=None):
    """A Box around a fitted model that bypasses Aspect preprocessing.

    ``reference`` stands in for retained training rows and ``candidates``
    for the rows prepared for prediction.
    """
    box = Box.__new__(Box)
    box.model = model if isinstance(model, Ensemble) else Ensemble(model)
    box.features = tuple(features)
    box.target = tuple(target)
    box.censor = None
    box.loss = None
    box._references = None
    box._validation_score = None
    box._fingerprint = None
    box._training_identity = None
    box.pipeline = SimpleNamespace(
        data_in=None,
        data_out=reference,
        column_transforms={},
    )
    box._pre_predict = lambda **kwargs: (candidates, box.pipeline)
    return box
