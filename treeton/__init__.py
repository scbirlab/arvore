from importlib.metadata import version

app_name = "duvidnn"
__author__ = "Eachan Johnson"
__version__ = version(app_name)

from .box import Box
from .mapping import ColumnMap
