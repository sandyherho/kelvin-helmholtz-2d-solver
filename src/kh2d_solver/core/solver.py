"""2D Kelvin-Helmholtz Instability Solver """

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import numba
from numba import jit, prange
import os


# Numba-optimized functions
@jit(nopython=True, parallel=True, cache=True)
def advect_upwind(q, u, w, dx, dz, dt):
    """Numba-optimized upwind advection."""
    nz, nx = q.shape
    qn = np.zeros_like(q)
    
    for i in prange(1, nz-1):
        for j in range(1, nx-1):
            # Upwind differencing
            if u[i, j] > 0:
                dudx = (q[i, j] - q[i, j-1]) / dx
            else:
                dudx = (q[i, j+1] - q[i, j]) / dx
            
            if w[i, j] > 0:
                dwdz = (q[i, j] - q[i-1, j]) / dz
            else:
                dwdz = (q[i+1, j] - q[i, j]) / dz
            
            qn[i, j] = q[i, j] - dt * (u[i, j] * dudx + w[i, j] * dwdz)
    
    # Boundary conditions
    qn[0, :] = q[0, :]
    qn[-1, :] = q[-1, :]
    qn[:, 0] = q[:, 0]
    qn[:, -1] = q[:, -1]
    
    return qn


@jit(nopython=True, parallel=True, cache=True)
def diffuse_explicit(q, nu, dx, dz, dt):
    """Numba-optimized explicit diffusion."""
    nz, nx = q.shape
    qn = np.zeros_like(q)
    alpha = nu * dt / dx**2
    beta = nu * dt / dz**2
    
    for i in prange(1, nz-1):
        for j in range(1, nx-1):
            qn[i, j] = q[i, j] + \
                alpha * (q[i, j+1] - 2*q[i, j] + q[i, j-1]) + \
                beta * (q[i+1, j] - 2*q[i, j] + q[i-1, j])
    
    # Boundary conditions
    qn[0, :] = q[0, :]
    qn[-1, :] = q[-1, :]
    qn[:, 0] = q[:, 0]
    qn[:, -1] = q[:, -1]
    
    return qn


@jit(nopython=True, parallel=True, cache=True)
def compute_divergence(u, w, dx, dz):
    """Numba-optimized divergence computation."""
    nz, nx = u.shape
    div = np.zeros((nz, nx))
    
    for i in prange(1, nz-1):
        for j in range(1, nx-1):
            div[i, j] = (u[i, j+1] - u[i, j-1]) / (2*dx) + \
                       (w[i+1, j] - w[i-1, j]) / (2*dz)
    
    return div


@jit(nopython=True, parallel=True, cache=True)
def compute_vorticity_z(u, w, dx, dz):
    """Numba-optimized z-component vorticity computation.
    ω_z = ∂w/∂x - ∂u/∂z
    """
    nz, nx = u.shape
    vort = np.zeros((nz, nx))
    
    for i in prange(1, nz-1):
        for j in range(1, nx-1):
            vort[i, j] = (w[i, j+1] - w[i, j-1]) / (2*dx) - \
                        (u[i+1, j] - u[i-1, j]) / (2*dz)
    
    return vort


@jit(nopython=True, parallel=True, cache=True)
def apply_pressure_gradient(u, w, p, dx, dz, dt):
    """Numba-optimized pressure gradient application."""
    nz, nx = u.shape
    u_new = u.copy()
    w_new = w.copy()
    
    for i in prange(1, nz-1):
        for j in range(1, nx-1):
            u_new[i, j] = u[i, j] - dt * (p[i, j+1] - p[i, j-1]) / (2*dx)
            w_new[i, j] = w[i, j] - dt * (p[i+1, j] - p[i-1, j]) / (2*dz)
    
    return u_new, w_new


@jit(nopython=True, parallel=True, cache=True)
def apply_buoyancy(w, rho, g_richardson, dt):
    """Numba-optimized buoyancy force."""
    nz, nx = w.shape
    w_new = w.copy()
    
    for i in prange(1, nz-1):
        for j in range(1, nx-1):
            w_new[i, j] = w[i, j] - dt * g_richardson * (rho[i, j] - 1.0)
    
    return w_new


class KH2DSolver:
    """Solver for 2D Kelvin-Helmholtz instability - Numba Optimized."""
    
    def __init__(
        self,
        nx: int = 256,
        nz: int = 128,
        lx: float = 2.0,
        lz: float = 1.0,
        verbose: bool = True,
        logger: Optional[Any] = None,
        n_cores: Optional[int] = None
    ):
        """Initialize the 2D KH solver with Numba optimization."""
        self.nx = nx
        self.nz = nz
        self.lx = lx
        self.lz = lz
        self.dx = lx / (nx - 1)
        self.dz = lz / (nz - 1)
        self.verbose = verbose
        self.logger = logger
        
        # Set number of cores for Numba
        if n_cores is None:
            n_cores = os.cpu_count()
        numba.set_num_threads(n_cores)
        
        self.x = np.linspace(0, lx, nx)
        self.z = np.linspace(0, lz, nz)
        self.X, self.Z = np.meshgrid(self.x, self.z)
        
        if verbose:
            print(f"  Solver initialized: {nx}x{nz} grid")
            print(f"  Using {n_cores} CPU cores for parallel computation")
    
    def _pressure_solve_fft(self, div):
        """Fast Poisson solver using FFT."""
        from scipy.fftpack import dst, idst
        
        rhs = -div[1:-1, 1:-1] * self.dx**2
        
        # 2D DST
        rhs_hat = dst(dst(rhs, type=1, axis=0), type=1, axis=1)
        
        m, n = rhs_hat.shape
        i, j = np.ogrid[1:m+1, 1:n+1]
        eigenvalues = 4 * (np.sin(np.pi*i/(2*(m+1)))**2 / self.dz**2 +
                          np.sin(np.pi*j/(2*(n+1)))**2 / self.dx**2)
        
        p_hat = rhs_hat / (eigenvalues * self.dx**2 + 1e-10)
        
        # Inverse transform
        p_inner = idst(idst(p_hat, type=1, axis=0), type=1, axis=1)
        p_inner /= (4 * m * n)
        
        p = np.zeros_like(div)
        p[1:-1, 1:-1] = p_inner
        
        return p
    
    def solve(
        self,
        u0: np.ndarray,
        w0: np.ndarray,
        rho0: np.ndarray,
        t_final: float = 10.0,
        dt: Optional[float] = None,
        n_snapshots: int = 100,
        reynolds: float = 1000,
        richardson: float = 0.25,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Solve the 2D Kelvin-Helmholtz instability problem - Numba Optimized."""
        # Make contiguous arrays for Numba
        u = np.ascontiguousarray(u0.copy())
        w = np.ascontiguousarray(w0.copy())
        rho = np.ascontiguousarray(rho0.copy())
        
        nu = 1.0 / reynolds
        kappa = nu
        g_richardson = 9.81 * richardson
        
        # Timestep calculation
        if dt is None:
            u_max = np.max(np.abs(u0)) + 1e-10
            w_max = np.max(np.abs(w0)) + 1e-10
            dt_cfl = 0.4 * min(self.dx/u_max, self.dz/w_max)
            dt_diff = 0.2 * min(self.dx**2, self.dz**2) / nu
            dt = min(dt_cfl, dt_diff)
            
            if self.verbose:
                print(f"  Auto dt = {dt:.3f} (CFL={dt_cfl:.3f}, Diff={dt_diff:.3f})")
        
        nt = int(t_final / dt)
        t_save = np.linspace(0, t_final, n_snapshots)
        save_idx = 0
        
        # Pre-allocate arrays
        u_hist = np.zeros((n_snapshots, self.nz, self.nx))
        w_hist = np.zeros((n_snapshots, self.nz, self.nx))
        rho_hist = np.zeros((n_snapshots, self.nz, self.nx))
        vort_hist = np.zeros((n_snapshots, self.nz, self.nx))
        
        u_hist[0] = u
        w_hist[0] = w
        rho_hist[0] = rho
        vort_hist[0] = compute_vorticity_z(u, w, self.dx, self.dz)
        save_idx = 1
        
        if show_progress:
            pbar = tqdm(range(nt), desc="Time integration")
        else:
            pbar = range(nt)
        
        t = 0
        for n in pbar:
            # Advection (Numba optimized)
            u = advect_upwind(u, u, w, self.dx, self.dz, dt)
            w = advect_upwind(w, u, w, self.dx, self.dz, dt)
            rho = advect_upwind(rho, u, w, self.dx, self.dz, dt)
            
            # Diffusion (Numba optimized)
            if nu * dt / self.dx**2 < 0.25 and nu * dt / self.dz**2 < 0.25:
                u = diffuse_explicit(u, nu, self.dx, self.dz, dt)
                w = diffuse_explicit(w, nu, self.dx, self.dz, dt)
                rho = diffuse_explicit(rho, kappa, self.dx, self.dz, dt)
            else:
                # Fall back to implicit for stability
                for _ in range(5):
                    u_old = u.copy()
                    w_old = w.copy()
                    rho_old = rho.copy()
                    alpha = 0.5 * nu * dt / self.dx**2
                    beta = 0.5 * nu * dt / self.dz**2
                    
                    u[1:-1, 1:-1] = (u[1:-1, 1:-1] + 
                        alpha * (u_old[1:-1, 2:] + u_old[1:-1, :-2]) +
                        beta * (u_old[2:, 1:-1] + u_old[:-2, 1:-1])) / \
                        (1 + 2*alpha + 2*beta)
                    w[1:-1, 1:-1] = (w[1:-1, 1:-1] + 
                        alpha * (w_old[1:-1, 2:] + w_old[1:-1, :-2]) +
                        beta * (w_old[2:, 1:-1] + w_old[:-2, 1:-1])) / \
                        (1 + 2*alpha + 2*beta)
                    rho[1:-1, 1:-1] = (rho[1:-1, 1:-1] + 
                        alpha * (rho_old[1:-1, 2:] + rho_old[1:-1, :-2]) +
                        beta * (rho_old[2:, 1:-1] + rho_old[:-2, 1:-1])) / \
                        (1 + 2*alpha + 2*beta)
            
            # Buoyancy (Numba optimized)
            w = apply_buoyancy(w, rho, g_richardson, dt)
            
            # Pressure correction
            div = compute_divergence(u, w, self.dx, self.dz)
            p = self._pressure_solve_fft(div / dt)
            
            # Velocity correction (Numba optimized)
            u, w = apply_pressure_gradient(u, w, p, self.dx, self.dz, dt)
            
            # Boundary conditions
            u[0, :] = u[-1, :] = 0
            w[0, :] = w[-1, :] = 0
            u[:, 0] = u[:, -1]  # Periodic in x
            w[:, 0] = w[:, -1]
            rho[:, 0] = rho[:, -1]
            
            # Check for stability
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                if self.verbose:
                    print(f"WARNING: Numerical instability at t={t:.3f}")
                break
            
            t += dt
            
            # Save snapshots
            if save_idx < n_snapshots and t >= t_save[save_idx]:
                u_hist[save_idx] = u
                w_hist[save_idx] = w
                rho_hist[save_idx] = rho
                vort_hist[save_idx] = compute_vorticity_z(u, w, self.dx, self.dz)
                save_idx += 1
        
        return {
            'x': self.x,
            'z': self.z,
            't': t_save[:save_idx],
            'u': u_hist[:save_idx],
            'w': w_hist[:save_idx],
            'rho': rho_hist[:save_idx],
            'vorticity': vort_hist[:save_idx],
            'params': {
                'reynolds': reynolds,
                'richardson': richardson,
                'dt': dt,
                'nu': nu
            }
        }
