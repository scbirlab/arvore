"""Composable prediction diagnostics."""

from .base import Requirements, ScoreContext, Scorer
from .registry import (
    SCORER_REGISTRY,
    register_scorer,
    resolve_scorers,
    scorer_name,
)
from .ensemble import EnsembleSD, TreeSD
from .leaves import InverseLeafOccupancy
from .distances import EuclideanDistance

__all__ = [
    "Requirements",
    "ScoreContext",
    "Scorer",
    "SCORER_REGISTRY",
    "register_scorer",
    "resolve_scorers",
    "scorer_name",
    "EnsembleSD",
    "TreeSD",
    "InverseLeafOccupancy",
    "EuclideanDistance",
]
