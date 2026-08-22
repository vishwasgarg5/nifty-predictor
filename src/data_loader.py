from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_PERIOD = "6mo"

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
]


# ============================================================
# SESSION
# ============================================================

def _session():
    """
    Create a curl_cffi session when available.

    GitHub Actions can sometimes have better compatibility with
    Yahoo Finance when curl_cffi impersonation is used.
    """

    try:

        from curl_cffi import requests as cffi_requests

        return cffi_requests.Session(
            impersonate="chrome"
        )

    except Exception as error:

        logger.debug(
            "curl_cffi session unavailable: %s",
            error,
        )

        return None


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def normalize_symbol(
    symbol: Any,
) -> str:
    """
    Normalize NSE symbols.

    Examples:

        RELIANCE     -> RELIANCE.NS
        RELIANCE.NS  -> RELIANCE.NS
        ^NSEI        -> ^NSEI
        TCS.BO       -> TCS.BO
    """

    if symbol is None:

        return ""

    value = str(symbol).strip().upper()

    if not value:

        return ""

    if value.startswith("^"):

        return value

    if value.endswith(".NS"):

        return value

    if value.endswith(".BO"):

        return value

    if "." in value:

        return value

    return f"{value}.NS"


# ============================================================
# DATA CLEANING
# ============================================================

def _clean_history(
    frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Clean a Yahoo Finance DataFrame.

    Handles:

        - None
        - empty DataFrames
        - MultiIndex columns
        - missing OHLC columns
        - invalid numeric values
        - duplicate timestamps
    """

    if frame is None:

        return pd.DataFrame()

    if not isinstance(
        frame,
        pd.DataFrame,
    ):

        return pd.DataFrame()

    if frame.empty:

        return pd.DataFrame()

    frame = frame.copy()

    # --------------------------------------------------------
    # Flatten MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        frame.columns,
        pd.MultiIndex,
    ):

        frame.columns = [
            str(column[0])
            for column in frame.columns
        ]

    # --------------------------------------------------------
    # Remove duplicate column names
    # --------------------------------------------------------

    frame = frame.loc[
        :,
        ~frame.columns.duplicated(),
    ]

    # --------------------------------------------------------
    # Ensure expected columns exist
    # --------------------------------------------------------

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:

        logger.debug(
            "Missing OHLC columns: %s",
            missing,
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Convert OHLCV columns to numeric
    # --------------------------------------------------------

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]:

        if column in frame.columns:

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    frame = frame.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ],
    )

    if frame.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Sort index
    # --------------------------------------------------------

    try:

        frame = frame.sort_index()

        frame = frame.loc[
            ~frame.index.duplicated(
                keep="last"
            )
        ]

    except Exception:

        pass

    return frame


# ============================================================
# YFINANCE TICKER HISTORY
# ============================================================

def _ticker_history(
    symbol: str,
    period: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch history using yf.Ticker().history().
    """

    session = _session()

    logger.debug(
        "Trying yf.Ticker history for %s",
        symbol,
    )

    if session is not None:

        ticker = yf.Ticker(
            symbol,
            session=session,
        )

    else:

        ticker = yf.Ticker(
            symbol
        )

    frame = ticker.history(
        period=period,
        auto_adjust=True,
        timeout=timeout,
        raise_errors=False,
    )

    return _clean_history(
        frame
    )


# ============================================================
# YFINANCE DOWNLOAD FALLBACK
# ============================================================

def _download_history(
    symbol: str,
    period: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch history using yf.download().

    This provides a second Yahoo Finance access path.
    """

    logger.debug(
        "Trying yf.download for %s",
        symbol,
    )

    frame = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=timeout,
    )

    return _clean_history(
        frame
    )


# ============================================================
# DATE RANGE FALLBACK
# ============================================================

def _date_range_history(
    symbol: str,
    days: int = 30,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch history using an explicit date range.

    This is useful when the period parameter fails.
    """

    end = datetime.utcnow()

    start = (
        end
        - timedelta(
            days=max(
                7,
                int(days),
            )
        )
    )

    session = _session()

    logger.debug(
        "Trying explicit date range for %s",
        symbol,
    )

    try:

        if session is not None:

            ticker = yf.Ticker(
                symbol,
                session=session,
            )

        else:

            ticker = yf.Ticker(
                symbol
            )

        frame = ticker.history(
            start=start.strftime(
                "%Y-%m-%d"
            ),
            end=(
                end
                + timedelta(days=1)
            ).strftime(
                "%Y-%m-%d"
            ),
            auto_adjust=True,
            timeout=timeout,
            raise_errors=False,
        )

        frame = _clean_history(
            frame
        )

        if not frame.empty:

            return frame

    except Exception as error:

        logger.debug(
            "Ticker date-range attempt failed "
            "for %s: %s",
            symbol,
            error,
        )

    try:

        frame = yf.download(
            tickers=symbol,
            start=start.strftime(
                "%Y-%m-%d"
            ),
            end=(
                end
                + timedelta(days=1)
            ).strftime(
                "%Y-%m-%d"
            ),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=timeout,
        )

        return _clean_history(
            frame
        )

    except Exception as error:

        logger.debug(
            "Download date-range attempt failed "
            "for %s: %s",
            symbol,
            error,
        )

        return pd.DataFrame()


# ============================================================
# MAIN HISTORY DOWNLOADER
# ============================================================

def download_history(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
) -> pd.DataFrame | None:
    """
    Download historical OHLCV market data.

    Fallback order:

        1. yf.Ticker().history()
        2. yf.download()
        3. Explicit date-range request

    Returns:

        Clean pandas DataFrame

    Returns None when all attempts fail.
    """

    symbol = normalize_symbol(
        symbol
    )

    if not symbol:

        logger.error(
            "Cannot download market data: "
            "empty symbol."
        )

        return None

    retries = max(
        1,
        int(retries),
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        retries + 1,
    ):

        logger.info(
            "Downloading market data | "
            "symbol=%s | "
            "attempt=%s/%s",
            symbol,
            attempt,
            retries,
        )

        # ----------------------------------------------------
        # METHOD 1
        # ----------------------------------------------------

        try:

            frame = _ticker_history(
                symbol=symbol,
                period=period,
            )

            if not frame.empty:

                logger.info(
                    "Market data loaded | "
                    "symbol=%s | "
                    "rows=%s | "
                    "source=yf.Ticker",
                    symbol,
                    len(frame),
                )

                return frame

        except Exception as error:

            last_error = error

            logger.warning(
                "yf.Ticker failed | "
                "symbol=%s | "
                "error=%s",
                symbol,
                error,
            )

        # ----------------------------------------------------
        # METHOD 2
        # ----------------------------------------------------

        try:

            frame = _download_history(
                symbol=symbol,
                period=period,
            )

            if not frame.empty:

                logger.info(
                    "Market data loaded | "
                    "symbol=%s | "
                    "rows=%s | "
                    "source=yf.download",
                    symbol,
                    len(frame),
                )

                return frame

        except Exception as error:

            last_error = error

            logger.warning(
                "yf.download failed | "
                "symbol=%s | "
                "error=%s",
                symbol,
                error,
            )

        # ----------------------------------------------------
        # METHOD 3
        # ----------------------------------------------------

        try:

            frame = _date_range_history(
                symbol=symbol,
                days=30,
            )

            if not frame.empty:

                logger.info(
                    "Market data loaded | "
                    "symbol=%s | "
                    "rows=%s | "
                    "source=date_range",
                    symbol,
                    len(frame),
                )

                return frame

        except Exception as error:

            last_error = error

            logger.warning(
                "Date-range fallback failed | "
                "symbol=%s | "
                "error=%s",
                symbol,
                error,
            )

        # ----------------------------------------------------
        # RETRY DELAY
        # ----------------------------------------------------

        if attempt < retries:

            delay = float(
                attempt * 2
            )

            logger.info(
                "Retrying %s after %.1f seconds.",
                symbol,
                delay,
            )

            time.sleep(
                delay
            )

    logger.error(
        "Could not load market data | "
        "symbol=%s | "
        "last_error=%s",
        symbol,
        last_error,
    )

    return None


# ============================================================
# STANDARD PIPELINE API
# ============================================================

def load_market_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Standard market data loader.

    This function is intended for the prediction pipeline.

    Raises RuntimeError if data cannot be loaded.
    """

    frame = download_history(
        symbol=symbol,
        period=period,
        retries=retries,
    )

    if frame is None or frame.empty:

        raise RuntimeError(
            f"Could not load market data "
            f"for {normalize_symbol(symbol)}."
        )

    return frame


def get_market_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Compatibility alias for load_market_data().
    """

    return load_market_data(
        symbol=symbol,
        period=period,
        retries=retries,
    )


def load_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Compatibility alias for load_market_data().
    """

    return load_market_data(
        symbol=symbol,
        period=period,
        retries=retries,
    )


def fetch_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Compatibility alias for load_market_data().
    """

    return load_market_data(
        symbol=symbol,
        period=period,
        retries=retries,
    )


# ============================================================
# NSE QUOTE FALLBACK
# ============================================================

def _nse_quote(
    symbol: str,
) -> dict[str, float | str] | None:
    """
    Get latest quote through nsepython.

    Used as a fallback for actual OHLC requests.
    """

    if symbol.startswith("^"):

        return None

    try:

        from nsepython import nse_eq

        clean = (
            symbol
            .replace(".NS", "")
            .replace(".BO", "")
        )

        quote = nse_eq(
            clean
        )

        if (
            not quote
            or "priceInfo" not in quote
        ):

            return None

        info = quote[
            "priceInfo"
        ]

        open_price = (
            info.get("open")
            or info.get("openingPrice")
        )

        close_price = (
            info.get("lastPrice")
            or info.get("close")
        )

        previous_close = (
            info.get("previousClose")
            or close_price
        )

        intraday = (
            info.get("intraDayHighLow")
            or {}
        )

        high_price = (
            intraday.get("max")
            if isinstance(
                intraday,
                dict,
            )
            else None
        )

        low_price = (
            intraday.get("min")
            if isinstance(
                intraday,
                dict,
            )
            else None
        )

        if (
            open_price is None
            or close_price is None
        ):

            return None

        return {
            "Open": float(
                open_price
            ),
            "High": float(
                high_price
                if high_price is not None
                else close_price
            ),
            "Low": float(
                low_price
                if low_price is not None
                else open_price
            ),
            "Close": float(
                close_price
            ),
            "prev_close": float(
                previous_close
            ),
            "source": "nse_eq",
        }

    except Exception as error:

        logger.debug(
            "NSE quote failed | "
            "symbol=%s | "
            "error=%s",
            symbol,
            error,
        )

        return None


# ============================================================
# ACTUAL OHLC
# ============================================================

def get_actual_ohlc(
    symbol: str,
    retries: int = 3,
) -> dict[str, float | str] | None:
    """
    Get the most recent OHLC data.

    Fallback order:

        1. Yahoo Finance
        2. NSE quote
    """

    symbol = normalize_symbol(
        symbol
    )

    # --------------------------------------------------------
    # YAHOO FINANCE
    # --------------------------------------------------------

    frame = download_history(
        symbol=symbol,
        period="5d",
        retries=retries,
    )

    if (
        frame is not None
        and not frame.empty
    ):

        last = frame.iloc[-1]

        if (
            pd.notna(
                last.get("Open")
            )
            and pd.notna(
                last.get("Close")
            )
        ):

            previous_close = (
                float(
                    frame["Close"].iloc[-2]
                )
                if len(frame) >= 2
                else float(
                    last["Close"]
                )
            )

            return {
                "Open": float(
                    last["Open"]
                ),
                "High": float(
                    last["High"]
                ),
                "Low": float(
                    last["Low"]
                ),
                "Close": float(
                    last["Close"]
                ),
                "prev_close": (
                    previous_close
                ),
                "source": "yfinance",
            }

    # --------------------------------------------------------
    # NSE FALLBACK
    # --------------------------------------------------------

    quote = _nse_quote(
        symbol
    )

    if quote is not None:

        return quote

    logger.error(
        "No actual OHLC data available "
        "for %s",
        symbol,
    )

    return None


# ============================================================
# ADDITIONAL COMPATIBILITY ALIASES
# ============================================================

def fetch_market_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Compatibility alias.
    """

    return load_market_data(
        symbol=symbol,
        period=period,
        retries=retries,
    )


def get_history(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Compatibility alias.
    """

    return load_market_data(
        symbol=symbol,
        period=period,
        retries=retries,
    )
