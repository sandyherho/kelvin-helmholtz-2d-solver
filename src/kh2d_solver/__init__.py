"""
Kelvin-Helmholtz 2D Instability Solver (Optimized)
===================================================

A high-performance solver for 2D Kelvin-Helmholtz instability with Numba acceleration.
"""

__version__ = "0.1.2"
__author__ = "Sandy H. S. Herho, Faiz R. Fajary, Iwan P. Anwar, Faruq Khadami, Gandhi Napitupulu, Nurjanna J. Trilaksono"
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
