import yaml
from pathlib import Path

def load_yaml(filepath: str) -> dict:
    """Load and parse a YAML file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {path.absolute()}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
