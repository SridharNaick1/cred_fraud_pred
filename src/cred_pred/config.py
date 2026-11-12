from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | Path = PROJECT_ROOT / "configs" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)