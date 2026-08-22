"""
Market Universe Loader.

Loads stock symbols from configured universe CSV files.

If the CSV files are not available, a built-in fallback universe
is used so the prediction pipeline can still run in GitHub Actions.

Supported universe names:

    nifty100
    nifty500

Optional IPO symbols can also be included.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.config import cfg


logger = logging.getLogger(__name__)


# ============================================================
# FALLBACK UNIVERSES
# ============================================================

_FALLBACKS: dict[str, list[str]] = {
    "nifty100": [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "LT.NS",
        "AXISBANK.NS",
    ],
    "nifty500": [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "LT.NS",
        "AXISBANK.NS",
        "KOTAKBANK.NS",
        "HINDUNILVR.NS",
        "MARUTI.NS",
        "SUNPHARMA.NS",
        "TITAN.NS",
        "BAJFINANCE.NS",
        "WIPRO.NS",
        "NTPC.NS",
        "POWERGRID.NS",
        "M&M.NS",
    ],
}


# ============================================================
# CONFIG HELPERS
# ============================================================

def get_config_value(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Safely get a configuration value.

    Supports both dictionary and attribute-style config objects.
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def normalize_symbol(
    symbol: Any,
) -> str:
    """
    Normalize a stock symbol for Yahoo Finance.

    Examples:

        RELIANCE      -> RELIANCE.NS
        reliance      -> RELIANCE.NS
        RELIANCE.NS   -> RELIANCE.NS
        ^NSEI         -> ^NSEI
    """

    if symbol is None:
        return ""

    value = str(symbol).strip().upper()

    if not value:
        return ""

    if value == "NAN":
        return ""

    if value.startswith("^"):
        return value

    if "." in value:
        return value

    return f"{value}.NS"


# ============================================================
# FILE LOADING
# ============================================================

def get_universes_directory() -> Path:
    """
    Get the configured universe directory.
    """

    paths = get_config_value(
        cfg,
        "paths",
        None,
    )

    directory = get_config_value(
        paths,
        "universes_dir",
        "data/universes",
    )

    path = Path(
        str(directory)
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT / path
        )

    return path


def load_universe_file(
    name: str,
) -> list[str]:
    """
    Load symbols from a universe CSV file.

    Expected location:

        data/universes/<name>.csv

    Supported column names:

        Symbol
        symbol
        Ticker
        ticker

    If none exist, the first column is used.
    """

    directory = (
        get_universes_directory()
    )

    path = (
        directory / f"{name}.csv"
    )

    if not path.exists():

        logger.info(
            "Universe file does not exist: %s",
            path,
        )

        return []

    try:

        frame = pd.read_csv(
            path
        )

    except Exception as error:

        logger.warning(
            "Could not read universe file %s: %s",
            path,
            error,
        )

        return []

    if frame.empty:

        logger.warning(
            "Universe file is empty: %s",
            path,
        )

        return []

    symbol_column = None

    for column in [
        "Symbol",
        "symbol",
        "Ticker",
        "ticker",
    ]:

        if column in frame.columns:

            symbol_column = column

            break

    if symbol_column is None:

        if len(frame.columns) == 0:

            logger.warning(
                "Universe file has no columns: %s",
                path,
            )

            return []

        symbol_column = (
            frame.columns[0]
        )

    symbols: list[str] = []

    for value in frame[
        symbol_column
    ].tolist():

        symbol = normalize_symbol(
            value
        )

        if symbol:

            symbols.append(
                symbol
            )

    symbols = sorted(
        set(symbols)
    )

    logger.info(
        "Loaded %s symbol(s) from %s",
        len(symbols),
        path,
    )

    return symbols


# ============================================================
# CONFIGURED UNIVERSE NAMES
# ============================================================

def get_configured_universe_names() -> list[str]:
    """
    Get configured primary and secondary universes.

    Example config:

        universes:
          primary:
            - nifty100

          secondary:
            - nifty500
    """

    universes = get_config_value(
        cfg,
        "universes",
        None,
    )

    if universes is None:

        return [
            "nifty100",
        ]

    primary = get_config_value(
        universes,
        "primary",
        [],
    )

    secondary = get_config_value(
        universes,
        "secondary",
        [],
    )

    if primary is None:
        primary = []

    if secondary is None:
        secondary = []

    if isinstance(
        primary,
        str,
    ):

        primary = [
            primary
        ]

    if isinstance(
        secondary,
        str,
    ):

        secondary = [
            secondary
        ]

    names: list[str] = []

    for name in list(
        primary
    ) + list(
        secondary
    ):

        value = str(
            name
        ).strip().lower()

        if value:

            names.append(
                value
            )

    if not names:

        names = [
            "nifty100",
        ]

    return names


# ============================================================
# IPO SUPPORT
# ============================================================

def get_ipo_symbols() -> list[str]:
    """
    Load eligible IPO symbols when enabled.

    Failure to load IPO data must never stop the
    main prediction pipeline.
    """

    universes = get_config_value(
        cfg,
        "universes",
        None,
    )

    include_ipo = get_config_value(
        universes,
        "include_ipo",
        False,
    )

    if not bool(include_ipo):

        return []

    try:

        from src.ipo import (
            filter_eligible_ipos,
        )

        symbols = (
            filter_eligible_ipos()
        )

        if symbols is None:

            return []

        result: list[str] = []

        for symbol in symbols:

            normalized = (
                normalize_symbol(
                    symbol
                )
            )

            if normalized:

                result.append(
                    normalized
                )

        logger.info(
            "Loaded %s eligible IPO symbol(s).",
            len(result),
        )

        return result

    except Exception as error:

        logger.warning(
            "Could not load eligible IPOs: %s",
            error,
        )

        return []


# ============================================================
# MAIN UNIVERSE FUNCTION
# ============================================================

def get_universe_symbols(
    names: list[str] | None = None,
) -> list[str]:
    """
    Return all symbols in the configured prediction universe.

    Loading order:

        1. Try configured CSV files.
        2. Use built-in fallback symbols if files are missing.
        3. Optionally add eligible IPO symbols.
        4. Remove duplicates.
        5. Return sorted symbols.

    This function is designed so that a missing universe CSV
    does NOT result in an empty prediction universe.
    """

    if names is None:

        names = (
            get_configured_universe_names()
        )

    if isinstance(
        names,
        str,
    ):

        names = [
            names
        ]

    symbols: set[str] = set()

    for raw_name in names:

        name = str(
            raw_name
        ).strip().lower()

        if not name:

            continue

        # ----------------------------------------------------
        # TRY CSV FILE
        # ----------------------------------------------------

        file_symbols = (
            load_universe_file(
                name
            )
        )

        if file_symbols:

            symbols.update(
                file_symbols
            )

            logger.info(
                "Universe %s loaded from CSV | "
                "symbols=%s",
                name,
                len(file_symbols),
            )

            continue

        # ----------------------------------------------------
        # USE FALLBACK
        # ----------------------------------------------------

        fallback_symbols = (
            _FALLBACKS.get(
                name,
                [],
            )
        )

        if fallback_symbols:

            symbols.update(
                fallback_symbols
            )

            logger.warning(
                "Universe %s file unavailable. "
                "Using fallback universe | "
                "symbols=%s",
                name,
                len(fallback_symbols),
            )

        else:

            logger.warning(
                "No CSV file or fallback available "
                "for universe: %s",
                name,
            )

    # --------------------------------------------------------
    # IPO SYMBOLS
    # --------------------------------------------------------

    ipo_symbols = (
        get_ipo_symbols()
    )

    if ipo_symbols:

        symbols.update(
            ipo_symbols
        )

    # --------------------------------------------------------
    # FINAL FALLBACK SAFETY
    # --------------------------------------------------------

    if not symbols:

        logger.warning(
            "Configured universe produced zero symbols. "
            "Using emergency Nifty fallback."
        )

        symbols.update(
            _FALLBACKS[
                "nifty100"
            ]
        )

    result = sorted(
        {
            normalize_symbol(symbol)
            for symbol in symbols
            if normalize_symbol(symbol)
        }
    )

    logger.info(
        "Total unique prediction symbols: %s",
        len(result),
    )

    return result


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def load_universe(
    names: list[str] | None = None,
) -> list[str]:
    """
    Compatibility alias.
    """

    return get_universe_symbols(
        names
    )


def get_universe(
    names: list[str] | None = None,
) -> list[str]:
    """
    Compatibility alias.
    """

    return get_universe_symbols(
        names
    )


# ============================================================
# CLI TEST
# ============================================================

def main() -> int:
    """
    Test universe loading.
    """

    symbols = (
        get_universe_symbols()
    )

    print()

    print("=" * 70)

    print("MARKET UNIVERSE")

    print("=" * 70)

    print(
        f"Total symbols: {len(symbols)}"
    )

    print()

    for symbol in symbols:

        print(symbol)

    print()

    return 0 if symbols else 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
