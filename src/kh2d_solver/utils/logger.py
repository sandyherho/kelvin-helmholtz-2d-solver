"""Logging System"""

import logging
from pathlib import Path
from datetime import datetime


class SimulationLogger:
    """Logger for KH2D simulations."""
    
    def __init__(self, scenario_name: str, log_dir: str = "logs", verbose: bool = True):
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{scenario_name}_{timestamp}.log"
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"kh2d_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def log_parameters(self, params: dict):
        self.info(f"Parameters: {params}")
    
    def log_timing(self, timing: dict):
        self.info(f"Timing: {timing}")
    
    def finalize(self):
        self.info(f"Simulation completed: {self.scenario_name}")
