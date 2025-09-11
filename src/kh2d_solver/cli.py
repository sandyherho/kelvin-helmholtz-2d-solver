#!/usr/bin/env python3
"""Command Line Interface for KH2D Solver"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from .core.solver import KH2DSolver
from .core.initial_conditions import get_initial_condition
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print professional header with credits."""
    print("\n" + "="*70)
    print(" "*15 + "KELVIN-HELMHOLTZ 2D INSTABILITY SOLVER")
    print(" "*25 + "Version 0.1.2")
    print("="*70)
    print("\nAuthors:")
    print("  • Sandy H. S. Herho (sandy.herho@email.ucr.edu)")
    print("  • Faiz R. Fajary")
    print("  • Iwan P. Anwar")
    print("  • Faruq Khadami")
    print("  • Gandhi Napitupulu")
    print("  • Nurjanna J. Trilaksono")
    print("\nLicense: WTFPL - Do What The F*** You Want To Public License")
    print("Repository: https://github.com/sandyherho/kelvin-helmholtz-2d-solver")
    print("="*70 + "\n")


def run_scenario(config: Dict[str, Any], output_dir: str = "outputs", 
                 verbose: bool = True, n_cores: int = None) -> None:
    """Run a single simulation scenario with enhanced output."""
    scenario_name = config.get('scenario_name', 'simulation')
    
    # Use cores from config if not specified in command line
    if n_cores is None:
        n_cores = config.get('n_cores', None)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*70}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger = SimulationLogger(
        scenario_name=scenario_name.lower().replace(' ', '_'),
        log_dir="logs",
        verbose=verbose
    )
    
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        # Print configuration
        if verbose:
            print("\nConfiguration Parameters:")
            print("-" * 40)
            for key, value in config.items():
                if key != 'scenario_name':
                    if isinstance(value, float):
                        print(f"  {key:20s}: {value:.3f}")
                    else:
                        print(f"  {key:20s}: {value}")
            print("-" * 40)
        
        # Initialize solver
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/5] Initializing Solver...")
            solver = KH2DSolver(
                nx=config.get('nx', 256),
                nz=config.get('nz', 128),
                lx=config.get('lx', 2.0),
                lz=config.get('lz', 1.0),
                verbose=verbose,
                logger=logger,
                n_cores=n_cores
            )
            if verbose:
                print(f"      ✓ Grid: {config.get('nx')}×{config.get('nz')}")
                print(f"      ✓ Domain: {config.get('lx'):.3f}×{config.get('lz'):.3f} m")
        
        # Set initial condition
        with timer.time_section("initial_condition"):
            if verbose:
                print("\n[2/5] Setting Initial Conditions...")
            ic_type = config.get('initial_condition', 'shear_layer')
            if verbose:
                print(f"      ✓ Type: {ic_type}")
                print(f"      ✓ Reynolds: {config.get('reynolds', 1000)}")
                print(f"      ✓ Richardson: {config.get('richardson', 0.25):.3f}")
            
            initial = get_initial_condition(ic_type, config)
            u0, w0, rho0 = initial(solver.x, solver.z)
            
            if verbose:
                print(f"      ✓ Max velocity: {np.max(np.abs(u0)):.3f} m/s")
                print(f"      ✓ Density range: [{np.min(rho0):.3f}, {np.max(rho0):.3f}] kg/m³")
        
        # Run simulation
        with timer.time_section("simulation"):
            if verbose:
                print("\n[3/5] Running Time Integration...")
                print(f"      Target time: {config.get('t_final', 10.0):.3f} s")
                print(f"      Snapshots: {config.get('n_snapshots', 100)}")
                print("\n" + "-"*40)
            
            result = solver.solve(
                u0=u0,
                w0=w0,
                rho0=rho0,
                t_final=config.get('t_final', 10.0),
                dt=config.get('dt', None),
                n_snapshots=config.get('n_snapshots', 100),
                reynolds=config.get('reynolds', 1000),
                richardson=config.get('richardson', 0.25),
                show_progress=verbose
            )
            
            if verbose:
                print("-"*40)
                print(f"      ✓ Integration completed")
                print(f"      ✓ Final time reached: {result['t'][-1]:.3f} s")
                print(f"      ✓ Max vorticity (ω_z): {np.max(np.abs(result['vorticity'])):.3f} s⁻¹")
        
        # Save results
        if config.get('save_netcdf', True):
            with timer.time_section("save_netcdf"):
                if verbose:
                    print("\n[4/5] Saving Data...")
                filename = f"{scenario_name.lower().replace(' ', '_')}.nc"
                DataHandler.save_netcdf(
                    filename=filename,
                    result=result,
                    metadata=config,
                    output_dir=output_dir
                )
                if verbose:
                    filepath = Path(output_dir) / filename
                    filesize = filepath.stat().st_size / (1024*1024)  # MB
                    print(f"      ✓ NetCDF saved: {filename} ({filesize:.3f} MB)")
        
        # Create animation
        if config.get('save_animation', True):
            with timer.time_section("create_animation"):
                if verbose:
                    print("\n[5/5] Creating Animation...")
                filename = f"{scenario_name.lower().replace(' ', '_')}.gif"
                Animator.create_gif(
                    result=result,
                    filename=filename,
                    output_dir=output_dir,
                    title=scenario_name,
                    fps=config.get('fps', 20),
                    dpi=config.get('dpi', 100)
                )
                if verbose:
                    filepath = Path(output_dir) / filename
                    filesize = filepath.stat().st_size / (1024*1024)  # MB
                    print(f"      ✓ Animation saved: {filename} ({filesize:.3f} MB)")
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            print("\n" + "="*70)
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print("-"*40)
            times = timer.get_times()
            print("Timing Breakdown:")
            for section, time_val in times.items():
                if section != 'total':
                    print(f"  {section:20s}: {time_val:6.3f} s")
            print("-"*40)
            print(f"  {'TOTAL TIME':20s}: {times['total']:6.3f} s")
            print("="*70)
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if verbose:
            print("\n" + "="*70)
            print("ERROR OCCURRED")
            print("-"*40)
            print(f"  {str(e)}")
            print("="*70 + "\n")
        raise
    
    finally:
        logger.finalize()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Kelvin-Helmholtz 2D Instability Solver (Numba Optimized)',
        epilog='For more information: https://github.com/sandyherho/kelvin-helmholtz-2d-solver'
    )
    
    parser.add_argument('scenario', nargs='?',
                       choices=['basic_shear', 'double_shear', 'rotating', 'forced'],
                       help='Predefined scenario to run')
    
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration file')
    
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all predefined scenarios')
    
    parser.add_argument('--output-dir', '-o', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    
    parser.add_argument('--cores', type=int, default=None,
                       help='Number of CPU cores to use (default: all available)')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        scenarios = sorted(configs_dir.glob('*.txt'))
        if verbose:
            print(f"Found {len(scenarios)} scenarios to run\n")
        for i, config_file in enumerate(scenarios, 1):
            if verbose:
                print(f"\n[{i}/{len(scenarios)}] Loading: {config_file.name}")
            config = ConfigManager.load(str(config_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
    elif args.scenario:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        config_file = configs_dir / f'{args.scenario}.txt'
        if config_file.exists():
            config = ConfigManager.load(str(config_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"Error: Configuration file not found: {config_file}")
            sys.exit(1)
    else:
        parser.print_help()
        if verbose:
            print("\nExample usage:")
            print("  kh2d-simulate basic_shear")
            print("  kh2d-simulate basic_shear --cores 8")
            print("  kh2d-simulate --config my_config.txt")
            print("  kh2d-simulate --all")
        sys.exit(0)
    
    if verbose:
        print("\nThank you for using KH2D Solver!")
        print("Contact: sandy.herho@email.ucr.edu")
        print("License: WTFPL - Do What The F*** You Want To Public License\n")


if __name__ == '__main__':
    main()
