"""Data Handling and Storage"""

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class DataHandler:
    """Handle data storage and retrieval for KH2D simulations."""
    
    @staticmethod
    def save_netcdf(filename: str, result: Dict[str, Any], 
                   metadata: Dict[str, Any], output_dir: str = "outputs") -> None:
        """Save simulation results to NetCDF file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            nc.createDimension('x', len(result['x']))
            nc.createDimension('z', len(result['z']))
            nc.createDimension('t', len(result['t']))
            
            nc_x = nc.createVariable('x', 'f4', ('x',))
            nc_z = nc.createVariable('z', 'f4', ('z',))
            nc_t = nc.createVariable('t', 'f4', ('t',))
            nc_u = nc.createVariable('u', 'f4', ('t', 'z', 'x'))
            nc_w = nc.createVariable('w', 'f4', ('t', 'z', 'x'))
            nc_rho = nc.createVariable('rho', 'f4', ('t', 'z', 'x'))
            nc_vort = nc.createVariable('vorticity_z', 'f4', ('t', 'z', 'x'))
            
            # Round to 3 decimal places before saving
            nc_x[:] = np.round(result['x'], 3)
            nc_z[:] = np.round(result['z'], 3)
            nc_t[:] = np.round(result['t'], 3)
            nc_u[:] = np.round(result['u'], 3)
            nc_w[:] = np.round(result['w'], 3)
            nc_rho[:] = np.round(result['rho'], 3)
            nc_vort[:] = np.round(result['vorticity'], 3)
            
            nc.description = "2D Kelvin-Helmholtz Instability Simulation"
            nc.created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            nc.software = "kh2d_solver v0.1.3"
            nc.authors = "Sandy H. S. Herho, Nurjanna J. Trilaksono, Faiz R. Fajary, Gandhi Napitupulu, Iwan P. Anwar, Faruq Khadami"
            nc.license = "WTFPL"
            
            if 'scenario_name' in metadata:
                nc.scenario = metadata['scenario_name']
            
            nc_x.units = "m"
            nc_z.units = "m"
            nc_t.units = "s"
            nc_u.units = "m/s"
            nc_w.units = "m/s"
            nc_rho.units = "kg/m^3"
            nc_vort.units = "1/s"
            nc_vort.description = "z-component of vorticity: omega_z = dw/dx - du/dz"
