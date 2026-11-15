"""Composable prediction diagnostics."""

from .base import Requirements, ScoreContext, Scorer
from .registry import (
    REFERENCE_TYPES,
    SCORER_REGISTRY,
    register_reference,
    register_scorer,
    resolve_scorers,
    scorer_name,
)
from .ensemble import EnsembleSD, TreeSD
from .leaves import InverseLeafOccupancy
from .distances import FeatureDistance
from .tanimoto import TanimotoDistance

__all__ = [
    "Requirements",
    "ScoreContext",
    "Scorer",
    "REFERENCE_TYPES",
    "SCORER_REGISTRY",
    "register_reference",
    "register_scorer",
    "resolve_scorers",
    "scorer_name",
    "EnsembleSD",
    "TreeSD",
    "InverseLeafOccupancy",
    "FeatureDistance",
    "TanimotoDistance",
]
