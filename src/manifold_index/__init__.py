"""
Refined Index Calculator
"""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("refined-index-calculator")
except PackageNotFoundError:
    __version__ = "1.0.3"
