"""Animation Creation"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Any

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'stix'


class Animator:
    """Create professional animations for KH2D solutions."""
    
    @staticmethod
    def create_gif(result: Dict[str, Any], filename: str, 
                  output_dir: str = "outputs", title: str = "KH Instability",
                  fps: int = 20, dpi: int = 100) -> None:
        """Create animated GIF of the simulation with professional layout."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x = result['x']
        z = result['z']
        t = result['t']
        vort = result['vorticity']
        rho = result['rho']
        params = result['params']
        
        # Create figure with better layout
        fig = plt.figure(figsize=(14, 10))
        
        # Create subplots with space for colorbars
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        
        # Adjust layout to make room for everything
        plt.subplots_adjust(left=0.1, right=0.85, top=0.92, bottom=0.08, hspace=0.35)
        
        # Calculate SYMMETRIC vorticity limits for consistent colorbar
        vmax = np.percentile(np.abs(vort), 95)
        vmax = np.ceil(vmax)
        vmin = -vmax
        
        rho_min, rho_max = np.min(rho), np.max(rho)
        
        # Create fixed colorbars
        levels_vort = np.linspace(vmin, vmax, 30)
        levels_rho = np.linspace(rho_min, rho_max, 30)
        
        # Initial plots to create colorbars
        im1 = ax1.contourf(x, z, vort[0], levels=levels_vort, 
                          cmap='RdBu_r', extend='both')
        im2 = ax2.contourf(x, z, rho[0], levels=levels_rho, 
                          cmap='viridis', extend='both')
        
        # Create colorbars once
        cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, fraction=0.05)
        cbar1.set_label(r'$\omega_z$ [s$^{-1}$]', fontsize=12, rotation=270, labelpad=20)
        cbar1.ax.tick_params(labelsize=10)
        
        # Set symmetric ticks for vorticity colorbar
        n_ticks = 5
        tick_values = np.linspace(vmin, vmax, n_ticks)
        cbar1.set_ticks(tick_values)
        cbar1.set_ticklabels([f'{v:.0f}' for v in tick_values])
        
        cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, fraction=0.05)
        cbar2.set_label(r'$\rho$ [kg/m$^3$]', fontsize=12, rotation=270, labelpad=20)
        cbar2.ax.tick_params(labelsize=10)
        
        # Format density colorbar
        n_density_ticks = 5
        density_tick_values = np.linspace(rho_min, rho_max, n_density_ticks)
        cbar2.set_ticks(density_tick_values)
        cbar2.set_ticklabels([f'{v:.3f}' for v in density_tick_values])
        
        def init():
            return []
        
        def animate(frame):
            # Clear only the plot areas, not the colorbars
            ax1.clear()
            ax2.clear()
            
            # Vorticity plot with corrected notation
            ax1.contourf(x, z, vort[frame], levels=levels_vort, 
                        cmap='RdBu_r', extend='both')
            ax1.set_ylabel(r'$z$ [m]', fontsize=14)
            ax1.set_title(r'Vorticity (z-component): $\omega_z = \partial w/\partial x - \partial u/\partial z$' + 
                         f' (t = {t[frame]:.3f} s)', fontsize=14)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            
            # Add contour lines for clarity
            ax1.contour(x, z, vort[frame], levels=10, colors='black', 
                       alpha=0.2, linewidths=0.5)
            
            # Density plot
            ax2.contourf(x, z, rho[frame], levels=levels_rho, 
                        cmap='viridis', extend='both')
            ax2.set_xlabel(r'$x$ [m]', fontsize=14)
            ax2.set_ylabel(r'$z$ [m]', fontsize=14)
            ax2.set_title(r'Density: $\rho$' + f' (t = {t[frame]:.3f} s)',
                         fontsize=14)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            
            # Add contour lines
            ax2.contour(x, z, rho[frame], levels=10, colors='white', 
                       alpha=0.2, linewidths=0.5)
            
            # Add frame counter
            ax1.text(0.02, 0.98, f'Frame: {frame+1}/{len(t)}', 
                    transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add simulation parameters
            param_text = (
                f'Re = {params["reynolds"]:.0f}  |  '
                f'Ri = {params["richardson"]:.3f}  |  '
                f'$\\Delta t$ = {params["dt"]:.3f} s  |  '
            )
            ax2.text(0.5, -0.15, param_text, transform=ax2.transAxes,
                    ha='center', va='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            return []
        
        # Create animation
        print(f"    Creating animation with {len(t)} frames...")
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), 
                           interval=1000/fps, blit=False)
        
        # Save animation
        print(f"    Saving animation to {filepath}...")
        anim.save(filepath, writer='pillow', fps=fps, dpi=dpi)
        plt.close()
        print(f"    Animation saved successfully!")
