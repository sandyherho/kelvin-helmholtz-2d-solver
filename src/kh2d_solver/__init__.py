"""
Kelvin-Helmholtz 2D Instability Solver (Optimized)
===================================================

A high-performance solver for 2D Kelvin-Helmholtz instability with Numba acceleration.
"""

__version__ = "0.1.4"
__author__ = "Sandy H. S. Herho, Nurjanna J. Trilaksono, Faiz R. Fajary, Gandhi Napitupulu, Iwan P. Anwar, Faruq Khadami"
__email__ = "sandy.herho@email.ucr.edu"
__license__ = "WTFPL"

from .core.solver import KH2DSolver
from .core.initial_conditions import (
    ShearLayer,
    DoubleShear,
    RotatingShear,
    ForcedTurbulence,
    get_initial_condition
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    "KH2DSolver",
    "ShearLayer",
    "DoubleShear",
    "RotatingShear",
    "ForcedTurbulence",
    "get_initial_condition",
    "ConfigManager",
    "DataHandler",
]
