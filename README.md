# `kh2d-solver`: Python-based incompressible Kelvin-Helmholtz 2D Instability Solver

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)
[![PyPI version](https://badge.fury.io/py/kh2d-solver.svg)](https://badge.fury.io/py/kh2d-solver)
[![DOI](https://zenodo.org/badge/1054548299.svg)](https://doi.org/10.5281/zenodo.17096830)

A high-performance solver for 2D Kelvin-Helmholtz instability for incompressible flows using finite difference methods with Numba acceleration.

## Features

- Numba JIT compilation for 10-50x speedup
- Multi-core parallelization support
- High-order finite difference schemes
- FFT-based Poisson solver
- Multiple predefined scenarios (shear layers, rotating flows, forced turbulence)
- NetCDF output format
- Animated GIF generation
- Conservation monitoring (mass, momentum, energy)

## Installation

### Install from PyPI (Recommended)

```bash
pip install kh2d-solver
```

### Install from Source

```bash
git clone https://github.com/sandyherho/kelvin-helmholtz-2d-solver.git
cd kelvin-helmholtz-2d-solver
pip install .
```

### Development Installation

For development with editable installation:
```bash
git clone https://github.com/sandyherho/kelvin-helmholtz-2d-solver.git
cd kelvin-helmholtz-2d-solver
pip install -e .
```

## Quick Start

Run simulations directly after installation:

```bash
# Run a basic shear layer simulation
kh2d-simulate basic_shear

# Run with multiple cores (e.g., 8 cores)
kh2d-simulate basic_shear --cores 8

# Run all predefined scenarios
kh2d-simulate --all

# Use a custom configuration
kh2d-simulate --config my_config.txt
```

### Python API Usage

```python
import numpy as np
from kh2d_solver import KH2DSolver, ShearLayer

# Create solver
solver = KH2DSolver(nx=256, nz=128, lx=2.0, lz=1.0)

# Set initial conditions
ic = ShearLayer(shear_thickness=0.05, u_top=1.0, u_bot=-1.0)
u0, w0, rho0 = ic(solver.x, solver.z)

# Run simulation
result = solver.solve(
    u0=u0, w0=w0, rho0=rho0,
    t_final=10.0,
    reynolds=1000,
    richardson=0.25
)

# Access results
vorticity = result['vorticity']
density = result['rho']
```

## Performance Options

The solver automatically uses Numba JIT compilation for critical loops. You can control parallelization:

```bash
# Use all available cores (default)
kh2d-simulate basic_shear

# Specify number of cores
kh2d-simulate basic_shear --cores 4

# Disable parallelization (single core)
kh2d-simulate basic_shear --cores 1
```

## Configuration

Configuration files are simple text files with key-value pairs:

```text
# Example configuration
scenario_name = My Simulation
nx = 256
nz = 128
t_final = 10.0
reynolds = 1000
richardson = 0.25
n_cores = 8  # Optional: specify cores (default: all available)
```

## Predefined Scenarios

1. **basic_shear** - Classical Kelvin-Helmholtz instability
2. **double_shear** - Double shear layer configuration
3. **rotating** - KH instability with system rotation
4. **forced** - Forced turbulence with energy injection

All scenarios use a unified 2.0 × 1.0 domain for direct comparison.

## Output

The solver generates:
- NetCDF files with complete flow field data
- Animated GIFs showing vorticity (ω_z) and density evolution
- Log files with simulation parameters and performance metrics

## Physics

The solver solves the 2D incompressible Navier-Stokes equations with density stratification:

- Continuity: ∇·**u** = 0
- Momentum: ∂**u**/∂t + (**u**·∇)**u** = -(1/ρ₀)∇p + ν∇²**u** - (ρ/ρ₀)g**k**
- Density: ∂ρ/∂t + **u**·∇ρ = κ∇²ρ

The vorticity shown is the z-component: ω_z = ∂w/∂x - ∂u/∂z

## Publishing to PyPI (For Maintainers)

### First-time Setup

1. **Create PyPI Account**: Register at [pypi.org](https://pypi.org)
2. **Get API Token**: Account Settings → API tokens → Create token
3. **Configure Poetry**:
```bash
poetry config pypi-token.pypi pypi-XXXXXXXX
```

### Publishing Process

```bash
# Update version in pyproject.toml
poetry version patch  # or minor/major

# Build the package
poetry build

# Publish to PyPI
poetry publish
```

### Version Management

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- netCDF4
- tqdm
- Numba

## Authors

- Sandy H. S. Herho (sandy.herho@email.ucr.edu)
- Nurjanna J. Trilaksono
- Faiz R. Fajary
- Gandhi Napitupulu
- Iwan P. Anwar
- Faruq Khadami

## License

This project is licensed under the WTFPL - Do What The F*ck You Want To Public License.
See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{kh2d_solver_2025,
  title = {{K}elvin-{H}elmholtz {2D} {I}nstability {S}olver},
  author = {Herho, Sandy H. S. and Trilaksono, Nurjanna J. and Fajary, 
  	   Faiz R. and Napitupulu, Gandhi and Anwar, Iwan P. and Khadami, Faruq},
  year = {2025},
  version = {0.1.4},
  url = {https://github.com/sandyherho/kelvin-helmholtz-2d-solver}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on [GitHub](https://github.com/sandyherho/kelvin-helmholtz-2d-solver/issues).
