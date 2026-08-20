import yaml
from pathlib import Path
from types import SimpleNamespace

def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)

cfg = load_config()
