from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


# ============================================================
# CONFIG CONVERSION
# ============================================================

def _to_ns(
    obj: Any,
) -> Any:
    """
    Recursively convert dictionaries into SimpleNamespace
    objects so configuration values can be accessed with:

        cfg.paths.universes_dir

    instead of:

        cfg["paths"]["universes_dir"]
    """

    if isinstance(
        obj,
        dict,
    ):

        return SimpleNamespace(
            **{
                key: _to_ns(value)
                for key, value in obj.items()
            }
        )

    if isinstance(
        obj,
        list,
    ):

        return [
            _to_ns(item)
            for item in obj
        ]

    return obj


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(
    path: str | Path | None = None,
) -> SimpleNamespace:
    """
    Load the project YAML configuration.

    Default location:

        config/config.yaml

    The default path is resolved relative to the project root,
    which makes it reliable in GitHub Actions.
    """

    if path is None:

        config_path = (
            PROJECT_ROOT
            / "config"
            / "config.yaml"
        )

    else:

        config_path = Path(
            path
        )

        if not config_path.is_absolute():

            config_path = (
                PROJECT_ROOT
                / config_path
            )

    if not config_path.exists():

        raise FileNotFoundError(
            "Configuration file not found: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(
            file
        )

    if data is None:

        data = {}

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "Configuration root must be "
            "a YAML mapping/dictionary."
        )

    return _to_ns(
        data
    )


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

cfg = load_config()
