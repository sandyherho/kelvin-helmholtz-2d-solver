"""Configuration Management"""

from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manage configuration files for KH2D simulations."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load configuration from TXT file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        return ConfigManager._parse_txt_config(path)
    
    @staticmethod
    def _parse_txt_config(filepath: Path) -> Dict[str, Any]:
        """Parse TXT configuration file."""
        config = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif '.' in value:
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                        else:
                            try:
                                value = int(value)
                            except ValueError:
                                pass
                    except (ValueError, AttributeError):
                        pass
                    
                    config[key] = value
        
        return config
