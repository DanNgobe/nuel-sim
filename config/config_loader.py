import yaml
import os
from typing import Dict, Any

_config = None

def load_config(config_path: str = None) -> Dict[str, Any]:
    global _config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "default.yaml")
    
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)
    return _config

def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        load_config()
    return _config