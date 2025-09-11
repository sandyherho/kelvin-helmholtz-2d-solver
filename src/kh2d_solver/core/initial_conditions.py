"""Initial Conditions for KH2D Simulations"""

import numpy as np
from typing import Tuple, Dict, Any


def get_initial_condition(ic_type: str, params: Dict[str, Any]):
    """Factory function for initial conditions."""
    if ic_type == 'shear_layer':
        return ShearLayer(**params)
    elif ic_type == 'double_shear':
        return DoubleShear(**params)
    elif ic_type == 'rotating':
        return RotatingShear(**params)
    elif ic_type == 'forced':
        return ForcedTurbulence(**params)
    else:
        return ShearLayer(**params)


class ShearLayer:
    """Basic Kelvin-Helmholtz shear layer."""
    
    def __init__(self, **kwargs):
        self.delta = kwargs.get('shear_thickness', 0.05)
        self.u_top = kwargs.get('u_top', 1.0)
        self.u_bot = kwargs.get('u_bot', -1.0)
        self.rho_top = kwargs.get('rho_top', 1.0)
        self.rho_bot = kwargs.get('rho_bot', 2.0)
        self.noise_amp = kwargs.get('noise_amplitude', 0.01)
        
    def __call__(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Z = np.meshgrid(x, z)
        lz = z[-1] - z[0]
        z_mid = lz / 2
        
        u = self.u_bot + (self.u_top - self.u_bot) * 0.5 * (1 + np.tanh((Z - z_mid) / self.delta))
        u += self.noise_amp * np.sin(4 * np.pi * X / (x[-1] - x[0])) * np.exp(-((Z - z_mid) / self.delta)**2)
        w = self.noise_amp * np.random.randn(*X.shape)
        rho = self.rho_bot + (self.rho_top - self.rho_bot) * 0.5 * (1 + np.tanh((Z - z_mid) / self.delta))
        
        return u, w, rho


class DoubleShear:
    """Double shear layer configuration."""
    
    def __init__(self, **kwargs):
        self.delta = kwargs.get('shear_thickness', 0.05)
        self.separation = kwargs.get('separation', 0.3)
        self.u_max = kwargs.get('u_max', 1.0)
        self.noise_amp = kwargs.get('noise_amplitude', 0.01)
        
    def __call__(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Z = np.meshgrid(x, z)
        lz = z[-1] - z[0]
        z1 = lz/2 - self.separation/2
        z2 = lz/2 + self.separation/2
        
        u = self.u_max * (np.tanh((Z - z1) / self.delta) - np.tanh((Z - z2) / self.delta) - 1)
        u += self.noise_amp * (np.sin(4 * np.pi * X / (x[-1] - x[0])) * 
                               (np.exp(-((Z - z1) / self.delta)**2) + np.exp(-((Z - z2) / self.delta)**2)))
        w = self.noise_amp * np.random.randn(*X.shape)
        rho = np.ones_like(X)
        rho[Z < z1] = 1.5
        rho[Z > z2] = 1.5
        
        return u, w, rho


class RotatingShear:
    """Shear layer with system rotation."""
    
    def __init__(self, **kwargs):
        self.base_shear = ShearLayer(**kwargs)
        self.rotation_rate = kwargs.get('rotation_rate', 0.5)
        
    def __call__(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, w, rho = self.base_shear(x, z)
        X, Z = np.meshgrid(x, z)
        u += self.rotation_rate * (Z - z[-1]/2)
        return u, w, rho


class ForcedTurbulence:
    """Forced turbulence with energy injection."""
    
    def __init__(self, **kwargs):
        self.base_shear = ShearLayer(**kwargs)
        self.n_modes = kwargs.get('n_forcing_modes', 5)
        self.force_amp = kwargs.get('forcing_amplitude', 0.1)
        
    def __call__(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, w, rho = self.base_shear(x, z)
        X, Z = np.meshgrid(x, z)
        lx, lz = x[-1] - x[0], z[-1] - z[0]
        
        for kx in range(1, self.n_modes + 1):
            for kz in range(1, self.n_modes + 1):
                phase = 2 * np.pi * np.random.rand()
                u += self.force_amp / np.sqrt(kx**2 + kz**2) * \
                     np.sin(2 * np.pi * kx * X / lx + phase) * \
                     np.cos(2 * np.pi * kz * Z / lz)
                w += self.force_amp / np.sqrt(kx**2 + kz**2) * \
                     np.cos(2 * np.pi * kx * X / lx) * \
                     np.sin(2 * np.pi * kz * Z / lz + phase)
        
        return u, w, rho
