"""Tree-based learning with reproducible data preparation."""
from importlib.metadata import version

app_name = "treeton"
__author__ = "Eachan Johnson"
__version__ = version(app_name)

from .box import Box
from .models.ensemble import Ensemble

__all__ = ["Box", "Ensemble"]
