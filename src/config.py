import yaml
from pathlib import Path
from types import SimpleNamespace

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v 
                              for k, v in raw.items()})

cfg = load_config()
