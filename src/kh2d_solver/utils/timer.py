"""Timing Utilities"""

import time
from typing import Dict
from contextlib import contextmanager


class Timer:
    """Timer class for tracking execution times."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str):
        self.start_times[name] = time.time()
    
    def stop(self, name: str):
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0
    
    @contextmanager
    def time_section(self, name: str):
        self.start(name)
        yield
        self.stop(name)
    
    def get_times(self) -> Dict[str, float]:
        return self.times.copy()
