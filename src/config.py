import yaml
from types import SimpleNamespace

def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(x) for x in obj]
    return obj

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return _to_ns(yaml.safe_load(f))

cfg = load_config()
