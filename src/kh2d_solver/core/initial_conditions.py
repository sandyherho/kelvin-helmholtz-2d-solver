"""Initial Conditions for KH2D Simulations

Coordinate System:
- x-axis: Horizontal (streamwise) direction, x ∈ [0, Lx]
- z-axis: Vertical direction with z=0 at BOTTOM (surface), positive UPWARD, z ∈ [0, Lz]
"""

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
    """Basic Kelvin-Helmholtz shear layer.
    
    Creates a horizontal shear layer at mid-height (z = Lz/2) with:
    - Faster flow (u_top) in the upper region (z > Lz/2)
    - Slower flow (u_bot) in the lower region (z < Lz/2)
    - Lighter fluid (rho_top) above
    - Heavier fluid (rho_bot) below
    """
    
    def __init__(self, **kwargs):
        self.delta = kwargs.get('shear_thickness', 0.05)
        self.u_top = kwargs.get('u_top', 1.0)
        self.u_bot = kwargs.get('u_bot', -1.0)
        self.rho_top = kwargs.get('rho_top', 1.0)  # Lighter fluid on top
        self.rho_bot = kwargs.get('rho_bot', 2.0)  # Heavier fluid at bottom
        self.noise_amp = kwargs.get('noise_amplitude', 0.01)
        
    def __call__(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate initial conditions.
        
        Args:
            x: x-coordinates (horizontal)
            z: z-coordinates (vertical, z=0 at bottom)
            
        Returns:
            u: Horizontal velocity
            w: Vertical velocity (positive upward)
            rho: Density field
        """
        X, Z = np.meshgrid(x, z)
        lz = z[-1] - z[0]
        z_mid = lz / 2  # Mid-height of domain
        
        # Horizontal velocity: transition from u_bot (bottom) to u_top (top)
        u = self.u_bot + (self.u_top - self.u_bot) * 0.5 * (1 + np.tanh((Z - z_mid) / self.delta))
        
        # Add perturbation to trigger instability
        u += self.noise_amp * np.sin(4 * np.pi * X / (x[-1] - x[0])) * np.exp(-((Z - z_mid) / self.delta)**2)
        
        # Small random vertical velocity
        w = self.noise_amp * np.random.randn(*X.shape)
        
        # Density: heavier fluid at bottom (stable stratification)
        rho = self.rho_bot + (self.rho_top - self.rho_bot) * 0.5 * (1 + np.tanh((Z - z_mid) / self.delta))
        
        return u, w, rho


class DoubleShear:
    """Double shear layer configuration.
    
    Creates two shear layers at z1 and z2 with a jet-like flow between them.
    """
    
    def __init__(self, **kwargs):
        self.delta = kwargs.get('shear_thickness', 0.05)
        self.separation = kwargs.get('separation', 0.3)
        self.u_max = kwargs.get('u_max', 1.0)
        self.noise_amp = kwargs.get('noise_amplitude', 0.01)
        
    def __call__(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate initial conditions with two shear layers."""
        X, Z = np.meshgrid(x, z)
        lz = z[-1] - z[0]
        
        # Two shear layers centered around mid-height
        z1 = lz/2 - self.separation/2  # Lower shear layer
        z2 = lz/2 + self.separation/2  # Upper shear layer
        
        # Jet-like flow between the two layers
        u = self.u_max * (np.tanh((Z - z1) / self.delta) - np.tanh((Z - z2) / self.delta) - 1)
        
        # Add perturbations at both shear layers
        u += self.noise_amp * (np.sin(4 * np.pi * X / (x[-1] - x[0])) * 
                               (np.exp(-((Z - z1) / self.delta)**2) + np.exp(-((Z - z2) / self.delta)**2)))
        
        w = self.noise_amp * np.random.randn(*X.shape)
        
        # Three-layer density structure
        rho = np.ones_like(X)
        rho[Z < z1] = 1.5  # Heavy fluid at bottom
        rho[Z > z2] = 1.5  # Heavy fluid at top
        # Light fluid in the middle (between z1 and z2)
        
        return u, w, rho


class RotatingShear:
    """Shear layer with system rotation.
    
    Adds a linear vertical shear due to rotation (Coriolis effect).
    """
    
    def __init__(self, **kwargs):
        self.base_shear = ShearLayer(**kwargs)
        self.rotation_rate = kwargs.get('rotation_rate', 0.5)
        
    def __call__(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate initial conditions with rotation effect."""
        u, w, rho = self.base_shear(x, z)
        X, Z = np.meshgrid(x, z)
        
        # Add linear shear due to rotation
        # Velocity increases linearly with height from bottom
        u += self.rotation_rate * (Z - z[-1]/2)
        
        return u, w, rho


class ForcedTurbulence:
    """Forced turbulence with energy injection.
    
    Adds multiple wave modes to create initial turbulent perturbations.
    """
    
    def __init__(self, **kwargs):
        self.base_shear = ShearLayer(**kwargs)
        self.n_modes = kwargs.get('n_forcing_modes', 5)
        self.force_amp = kwargs.get('forcing_amplitude', 0.1)
        
    def __call__(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate initial conditions with forced turbulence."""
        u, w, rho = self.base_shear(x, z)
        X, Z = np.meshgrid(x, z)
        lx, lz = x[-1] - x[0], z[-1] - z[0]
        
        # Add multiple wave modes for initial turbulence
        for kx in range(1, self.n_modes + 1):
            for kz in range(1, self.n_modes + 1):
                phase = 2 * np.pi * np.random.rand()
                
                # Add horizontal velocity perturbations
                u += self.force_amp / np.sqrt(kx**2 + kz**2) * \
                     np.sin(2 * np.pi * kx * X / lx + phase) * \
                     np.cos(2 * np.pi * kz * Z / lz)
                
                # Add vertical velocity perturbations (positive upward)
                w += self.force_amp / np.sqrt(kx**2 + kz**2) * \
                     np.cos(2 * np.pi * kx * X / lx) * \
                     np.sin(2 * np.pi * kz * Z / lz + phase)
        
        return u, w, rho
