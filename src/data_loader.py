import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


# ============================================================
# SESSION
# ============================================================

def _session():
    """
    Create a curl_cffi session when available.

    This can improve Yahoo Finance reliability,
    especially in GitHub Actions.
    """

    try:
        from curl_cffi import requests as cffi_requests

        return cffi_requests.Session(
            impersonate="chrome"
        )

    except Exception:

        return None


# ============================================================
# YAHOO HISTORY DOWNLOAD
# ============================================================

def download_history(
    symbol: str,
    period: str = "6mo",
    retries: int = 3,
) -> pd.DataFrame | None:
    """
    Download historical OHLCV data from Yahoo Finance.

    Returns a DataFrame or None.
    """

    session = _session()

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            logger.info(
                "Downloading Yahoo history | "
                "symbol=%s | attempt=%s/%s",
                symbol,
                attempt,
                retries,
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

            # ------------------------------------------------
            # PRIMARY DOWNLOAD
            # ------------------------------------------------

            frame = ticker.history(
                period=period,
                auto_adjust=True,
                timeout=30,
            )

            # ------------------------------------------------
            # DATE RANGE FALLBACK
            # ------------------------------------------------

            if frame is None or frame.empty:

                end = datetime.utcnow()

                start = (
                    end
                    - timedelta(
                        days=220
                    )
                )

                logger.warning(
                    "Period download empty for %s. "
                    "Trying date range.",
                    symbol,
                )

                frame = ticker.history(
                    start=start.strftime(
                        "%Y-%m-%d"
                    ),
                    end=end.strftime(
                        "%Y-%m-%d"
                    ),
                    auto_adjust=True,
                    timeout=30,
                )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if frame is None or frame.empty:

                raise ValueError(
                    "Yahoo returned empty history."
                )

            frame = frame.copy()

            # Flatten MultiIndex columns if Yahoo returns them.
            if isinstance(
                frame.columns,
                pd.MultiIndex,
            ):

                frame.columns = (
                    frame.columns
                    .get_level_values(0)
                )

            # Standardize column names.
            frame.columns = [
                str(column).strip()
                for column in frame.columns
            ]

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
            ]

            missing = [
                column
                for column in required_columns
                if column not in frame.columns
            ]

            if missing:

                raise ValueError(
                    "Missing required columns: "
                    + ", ".join(missing)
                )

            frame = frame.dropna(
                subset=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            )

            if frame.empty:

                raise ValueError(
                    "No valid OHLC rows."
                )

            frame = frame.sort_index()

            logger.info(
                "Yahoo history loaded | "
                "symbol=%s | rows=%s",
                symbol,
                len(frame),
            )

            return frame

        except Exception as error:

            last_error = error

            logger.warning(
                "Yahoo download failed | "
                "symbol=%s | attempt=%s/%s | error=%s",
                symbol,
                attempt,
                retries,
                error,
            )

            if attempt < retries:

                time.sleep(
                    float(attempt) * 2.0
                )

    logger.error(
        "Yahoo history failed for %s: %s",
        symbol,
        last_error,
    )

    return None


# ============================================================
# NSE LIVE QUOTE
# ============================================================

def _nse_quote(
    symbol: str,
) -> dict | None:
    """
    Fetch the latest quote from NSE using nsepython.

    Used as a fallback when Yahoo data is unavailable.
    """

    if symbol.startswith("^"):

        return None

    try:

        from nsepython import nse_eq

        clean_symbol = (
            symbol
            .replace(".NS", "")
            .replace(".BO", "")
        )

        quote = nse_eq(
            clean_symbol
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
                if previous_close is not None
                else close_price
            ),
            "source": "nse_eq",
        }

    except Exception as error:

        logger.debug(
            "NSE quote failed for %s: %s",
            symbol,
            error,
        )

        return None


# ============================================================
# NSE HISTORY FALLBACK
# ============================================================

def _nse_history_quiet(
    symbol: str,
    days: int = 30,
) -> pd.DataFrame | None:
    """
    Attempt to retrieve historical NSE data.

    This is a fallback only.
    """

    if symbol.startswith("^"):

        return None

    try:

        import logging as _logging

        _logging.getLogger(
            "nsepython"
        ).setLevel(
            _logging.ERROR
        )

        from nsepython import equity_history

        clean_symbol = (
            symbol
            .replace(".NS", "")
            .replace(".BO", "")
        )

        end = datetime.now()

        start = (
            end
            - timedelta(
                days=days
            )
        )

        raw = equity_history(
            clean_symbol,
            "EQ",
            start.strftime(
                "%d-%m-%Y"
            ),
            end.strftime(
                "%d-%m-%Y"
            ),
        )

        if raw is None:

            return None

        if isinstance(
            raw,
            dict,
        ):

            if (
                "data" not in raw
                or not raw["data"]
            ):

                return None

            raw = pd.DataFrame(
                raw["data"]
            )

        if (
            not isinstance(
                raw,
                pd.DataFrame,
            )
            or raw.empty
        ):

            return None

        column_map = {}

        for column in raw.columns:

            column_lower = (
                str(column)
                .lower()
            )

            if "open" in column_lower:

                column_map[column] = "Open"

            elif "high" in column_lower:

                column_map[column] = "High"

            elif "low" in column_lower:

                column_map[column] = "Low"

            elif (
                "close" in column_lower
                and "prev" not in column_lower
            ):

                column_map[column] = "Close"

            elif "vol" in column_lower:

                column_map[column] = "Volume"

        frame = raw.rename(
            columns=column_map
        )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
        ]

        if not all(
            column in frame.columns
            for column in required
        ):

            return None

        frame = frame.dropna(
            subset=required
        )

        if frame.empty:

            return None

        return frame

    except Exception as error:

        logger.debug(
            "NSE history failed for %s: %s",
            symbol,
            error,
        )

        return None


# ============================================================
# MARKET DATA LOADER
# ============================================================

def load_market_data(
    symbol: str,
    period: str = "6mo",
    retries: int = 3,
) -> pd.DataFrame:
    """
    Main market data interface.

    This function is used directly by:

        src.prediction_pipeline

    Returns historical OHLCV data.

    Priority:

        1. Yahoo Finance
        2. NSE historical data

    Raises RuntimeError when no data is available.
    """

    # --------------------------------------------------------
    # YAHOO FINANCE
    # --------------------------------------------------------

    frame = download_history(
        symbol=symbol,
        period=period,
        retries=retries,
    )

    if (
        frame is not None
        and not frame.empty
    ):

        logger.info(
            "Market data loaded from Yahoo | "
            "symbol=%s | rows=%s",
            symbol,
            len(frame),
        )

        return frame

    # --------------------------------------------------------
    # NSE HISTORY FALLBACK
    # --------------------------------------------------------

    logger.warning(
        "Yahoo unavailable for %s. "
        "Trying NSE historical fallback.",
        symbol,
    )

    frame = _nse_history_quiet(
        symbol=symbol,
        days=220,
    )

    if (
        frame is not None
        and not frame.empty
    ):

        logger.info(
            "Market data loaded from NSE | "
            "symbol=%s | rows=%s",
            symbol,
            len(frame),
        )

        return frame

    # --------------------------------------------------------
    # FAILURE
    # --------------------------------------------------------

    raise RuntimeError(
        f"Could not load market data for {symbol} "
        "from Yahoo Finance or NSE."
    )


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def load_data(
    symbol: str,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Compatibility interface.
    """

    return load_market_data(
        symbol=symbol,
        period=period,
    )


def get_data(
    symbol: str,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Compatibility interface.
    """

    return load_market_data(
        symbol=symbol,
        period=period,
    )


def fetch_data(
    symbol: str,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Compatibility interface.
    """

    return load_market_data(
        symbol=symbol,
        period=period,
    )


def fetch_market_data(
    symbol: str,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Compatibility interface.
    """

    return load_market_data(
        symbol=symbol,
        period=period,
    )


# ============================================================
# ACTUAL OHLC
# ============================================================

def get_actual_ohlc(
    symbol: str,
    retries: int = 3,
) -> dict | None:
    """
    Return the latest available OHLC values.

    Priority:

        1. Yahoo Finance
        2. NSE live quote
        3. NSE historical data
    """

    # --------------------------------------------------------
    # YAHOO
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

        if isinstance(
            frame.columns,
            pd.MultiIndex,
        ):

            frame.columns = (
                frame.columns
                .get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
        ]

        if all(
            column in frame.columns
            for column in required
        ):

            last = frame.iloc[-1]

            if (
                pd.notna(last["Close"])
                and pd.notna(last["Open"])
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
                    "prev_close": previous_close,
                    "source": "yfinance",
                }

    # --------------------------------------------------------
    # NSE LIVE QUOTE
    # --------------------------------------------------------

    quote = _nse_quote(
        symbol
    )

    if quote:

        return quote

    # --------------------------------------------------------
    # NSE HISTORY
    # --------------------------------------------------------

    frame = _nse_history_quiet(
        symbol=symbol,
        days=30,
    )

    if (
        frame is not None
        and not frame.empty
    ):

        last = frame.iloc[-1]

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
            "prev_close": previous_close,
            "source": "nsepython",
        }

    logger.error(
        "No actual data available for %s",
        symbol,
    )

    return None


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    test_symbol = "RELIANCE.NS"

    print()

    print("=" * 70)
    print("MARKET DATA LOADER TEST")
    print("=" * 70)

    try:

        data = load_market_data(
            test_symbol
        )

        print(
            f"Symbol: {test_symbol}"
        )

        print(
            f"Rows: {len(data)}"
        )

        print()

        print(
            data.tail().to_string()
        )

    except Exception as error:

        print(
            f"FAILED: {error}"
        )
