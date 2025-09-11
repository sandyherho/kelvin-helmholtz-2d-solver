# `kh2d`: Python-based Kelvin-Helmholtz 2D Instability Solver

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)

A high-performance solver for 2D Kelvin-Helmholtz instability using finite difference methods with Numba acceleration.

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

### Prerequisites

Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install the Package

```bash
git clone https://github.com/sandyherho/kelvin-helmholtz-2d-solver.git
cd kelvin-helmholtz-2d-solver
poetry install
```

## Quick Start

Activate the poetry environment and run simulations:

```bash
# Activate poetry shell
poetry shell

# Run a basic shear layer simulation
kh2d-simulate basic_shear

# Run with multiple cores (e.g., 8 cores)
kh2d-simulate basic_shear --cores 8

# Run all predefined scenarios
kh2d-simulate --all

# Use a custom configuration
kh2d-simulate --config my_config.txt
```

Or run without activating the shell:

```bash
# Run directly with poetry
poetry run kh2d-simulate basic_shear
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

Configuration files are simple text files with key-value pairs, located in the `configs/` directory:

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

The vorticity shown is the z-component: $ω_z$ = ∂w/∂x - ∂u/∂z


## Authors

- Sandy H. S. Herho (sandy.herho@email.ucr.edu)
- Faiz R. Fajary
- Iwan P. Anwar
- Faruq Khadami
- Gandhi Napitupulu
- Nurjanna J. Trilaksono

## License

This project is licensed under the WTFPL - Do What The F*ck You Want To Public License.
See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{kh2d_solver_2025,
  title = {Kelvin-Helmholtz 2D Instability Solver},
  author = {Herho, Sandy H. S. and Fajary, Faiz R. and Anwar, Iwan P. and 
           Khadami, Faruq and Napitupulu, Gandhi and Trilaksono, Nurjanna J.},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/sandyherho/kelvin-helmholtz-2d-solver}
}
```
