#!/usr/bin/env python3

"""
Production Circuit Breaker.

Controls whether stock predictions are allowed to be sent
to Telegram.

States
------
CLOSED
    System is healthy.
    Predictions are allowed.

OPEN
    System is unhealthy or a critical failure occurred.
    Predictions are blocked.

HALF_OPEN
    Cooldown period has passed.
    The system may perform a limited recovery check.

The breaker state is persisted to:

    data/monitoring/circuit_breaker.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("circuit_breaker")


# ============================================================
# STATES
# ============================================================

CLOSED = "CLOSED"
OPEN = "OPEN"
HALF_OPEN = "HALF_OPEN"

VALID_STATES = {
    CLOSED,
    OPEN,
    HALF_OPEN,
}


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    "enabled": True,
    "failure_threshold": 1,
    "cooldown_minutes": 60,
    "health_open_threshold": 50,
    "health_close_threshold": 80,
    "state_file": "data/monitoring/circuit_breaker.json",
}


# ============================================================
# CONFIG HELPERS
# ============================================================

def _object_to_dict(value: Any) -> dict[str, Any]:
    """
    Convert a config object or mapping into a dictionary.
    """

    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "items"):
        try:
            return dict(value.items())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return {}


def load_circuit_breaker_config() -> dict[str, Any]:
    """
    Load circuit breaker configuration.

    Reads:

        circuit_breaker:
          ...

    from src.config.cfg.

    Falls back to safe defaults if the project config
    cannot be imported.
    """

    config = dict(DEFAULT_CONFIG)

    try:
        from src.config import cfg

        section = getattr(
            cfg,
            "circuit_breaker",
            None,
        )

        values = _object_to_dict(section)

        if values:
            config.update(values)

    except Exception as error:

        logger.warning(
            "Could not load circuit breaker config. "
            "Using defaults: %s",
            error,
        )

    return config


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_project_path(
    value: str | Path,
) -> Path:
    """
    Resolve a path relative to the project root.
    """

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_state_file_path() -> Path:
    """
    Return the circuit breaker state file path.
    """

    config = load_circuit_breaker_config()

    return resolve_project_path(
        config.get(
            "state_file",
            DEFAULT_CONFIG["state_file"],
        )
    )


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:
    """
    Return the current UTC time.
    """

    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    """
    Return the current UTC time in ISO format.
    """

    return utc_now().isoformat()


def parse_datetime(
    value: Any,
) -> datetime | None:
    """
    Safely parse an ISO datetime.
    """

    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if not value:
        return None

    try:

        parsed = datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00",
            )
        )

        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )

    except Exception:

        return None


# ============================================================
# DEFAULT STATE
# ============================================================

def default_state() -> dict[str, Any]:
    """
    Return a new healthy circuit breaker state.
    """

    now = utc_now_iso()

    return {
        "state": CLOSED,
        "failure_count": 0,
        "opened_at": None,
        "last_failure_at": None,
        "last_success_at": now,
        "last_health_score": None,
        "last_health_status": "UNKNOWN",
        "reason": None,
        "updated_at": now,
    }


# ============================================================
# STATE VALIDATION
# ============================================================

def normalize_state(
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Validate and normalize circuit breaker state.
    """

    defaults = default_state()

    if not isinstance(state, dict):
        return defaults

    result = dict(defaults)

    result.update(state)

    current_state = str(
        result.get(
            "state",
            CLOSED,
        )
    ).upper()

    if current_state not in VALID_STATES:

        logger.warning(
            "Invalid circuit breaker state '%s'. "
            "Resetting to CLOSED.",
            current_state,
        )

        current_state = CLOSED

    result["state"] = current_state

    try:

        result["failure_count"] = max(
            0,
            int(
                result.get(
                    "failure_count",
                    0,
                )
            ),
        )

    except Exception:

        result["failure_count"] = 0

    result["updated_at"] = utc_now_iso()

    return result


# ============================================================
# LOAD / SAVE STATE
# ============================================================

def load_state() -> dict[str, Any]:
    """
    Load persisted circuit breaker state.

    If no state file exists, a healthy CLOSED state is returned.
    """

    path = get_state_file_path()

    if not path.exists():

        return default_state()

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return normalize_state(data)

    except Exception as error:

        logger.error(
            "Could not read circuit breaker state: %s",
            error,
        )

        return default_state()


def save_state(
    state: dict[str, Any],
) -> Path:
    """
    Save circuit breaker state atomically.
    """

    path = get_state_file_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized = normalize_state(
        state
    )

    normalized["updated_at"] = (
        utc_now_iso()
    )

    temporary_path: Path | None = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            suffix=".tmp",
        ) as temporary_file:

            json.dump(
                normalized,
                temporary_file,
                indent=2,
                default=str,
            )

            temporary_path = Path(
                temporary_file.name
            )

        os.replace(
            temporary_path,
            path,
        )

        return path

    except Exception:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            try:
                temporary_path.unlink()
            except Exception:
                pass

        raise


# ============================================================
# COOLDOWN
# ============================================================

def cooldown_expired(
    state: dict[str, Any],
    cooldown_minutes: int,
) -> bool:
    """
    Return True if the OPEN state has completed its cooldown.
    """

    opened_at = parse_datetime(
        state.get(
            "opened_at"
        )
    )

    if opened_at is None:

        return True

    cooldown = timedelta(
        minutes=max(
            0,
            int(cooldown_minutes),
        )
    )

    return utc_now() >= (
        opened_at + cooldown
    )


# ============================================================
# STATE TRANSITIONS
# ============================================================

def open_circuit(
    reason: str,
    health_score: float | None = None,
    health_status: str | None = None,
) -> dict[str, Any]:
    """
    Open the circuit breaker.

    Predictions will be blocked.
    """

    state = load_state()

    state["state"] = OPEN

    state["failure_count"] = max(
        1,
        int(
            state.get(
                "failure_count",
                0,
            )
        ),
    )

    state["opened_at"] = utc_now_iso()

    state["last_failure_at"] = (
        utc_now_iso()
    )

    state["reason"] = str(reason)

    if health_score is not None:

        try:

            state["last_health_score"] = (
                float(health_score)
            )

        except Exception:

            pass

    if health_status is not None:

        state["last_health_status"] = str(
            health_status
        ).upper()

    save_state(state)

    logger.warning(
        "Circuit breaker OPENED: %s",
        reason,
    )

    return load_state()


def half_open_circuit() -> dict[str, Any]:
    """
    Move the breaker into HALF_OPEN recovery mode.
    """

    state = load_state()

    state["state"] = HALF_OPEN

    state["reason"] = (
        "Cooldown completed. "
        "Waiting for health recovery confirmation."
    )

    save_state(state)

    logger.info(
        "Circuit breaker moved to HALF_OPEN."
    )

    return load_state()


def close_circuit(
    reason: str = "System health recovered.",
    health_score: float | None = None,
    health_status: str | None = "HEALTHY",
) -> dict[str, Any]:
    """
    Close the circuit breaker.

    Predictions are allowed again.
    """

    state = load_state()

    state["state"] = CLOSED

    state["failure_count"] = 0

    state["opened_at"] = None

    state["reason"] = str(reason)

    state["last_success_at"] = (
        utc_now_iso()
    )

    if health_score is not None:

        try:

            state["last_health_score"] = (
                float(health_score)
            )

        except Exception:

            pass

    if health_status is not None:

        state["last_health_status"] = str(
            health_status
        ).upper()

    save_state(state)

    logger.info(
        "Circuit breaker CLOSED: %s",
        reason,
    )

    return load_state()


# ============================================================
# FAILURE / SUCCESS REGISTRATION
# ============================================================

def register_failure(
    reason: str,
    health_score: float | None = None,
    health_status: str | None = None,
) -> dict[str, Any]:
    """
    Register a system failure.

    Opens the circuit once the configured failure threshold
    has been reached.
    """

    config = load_circuit_breaker_config()

    if not bool(
        config.get(
            "enabled",
            True,
        )
    ):

        return load_state()

    state = load_state()

    state["failure_count"] = (
        int(
            state.get(
                "failure_count",
                0,
            )
        )
        + 1
    )

    state["last_failure_at"] = (
        utc_now_iso()
    )

    state["reason"] = str(reason)

    if health_score is not None:

        try:

            state["last_health_score"] = (
                float(health_score)
            )

        except Exception:

            pass

    if health_status is not None:

        state["last_health_status"] = str(
            health_status
        ).upper()

    threshold = max(
        1,
        int(
            config.get(
                "failure_threshold",
                1,
            )
        ),
    )

    if state["failure_count"] >= threshold:

        state["state"] = OPEN

        state["opened_at"] = (
            utc_now_iso()
        )

        logger.warning(
            "Circuit breaker opening after %s failure(s).",
            state["failure_count"],
        )

    save_state(state)

    return load_state()


def register_success(
    health_score: float | None = None,
    health_status: str | None = "HEALTHY",
) -> dict[str, Any]:
    """
    Register a successful health check.

    The circuit closes only when the health score meets
    the configured recovery threshold.
    """

    config = load_circuit_breaker_config()

    state = load_state()

    close_threshold = float(
        config.get(
            "health_close_threshold",
            DEFAULT_CONFIG[
                "health_close_threshold"
            ],
        )
    )

    numeric_score: float | None = None

    if health_score is not None:

        try:

            numeric_score = float(
                health_score
            )

            state["last_health_score"] = (
                numeric_score
            )

        except Exception:

            numeric_score = None

    if health_status is not None:

        state["last_health_status"] = str(
            health_status
        ).upper()

    state["last_success_at"] = (
        utc_now_iso()
    )

    if (
        numeric_score is not None
        and numeric_score >= close_threshold
    ):

        return close_circuit(
            reason=(
                "Health score recovered to "
                f"{numeric_score:.2f}."
            ),
            health_score=numeric_score,
            health_status=health_status,
        )

    save_state(state)

    return load_state()


# ============================================================
# MONITORING INTEGRATION
# ============================================================

def update_from_health(
    health_score: Any,
    health_status: Any,
    reason: str | None = None,
) -> dict[str, Any]:
    """
    Update the circuit breaker from production monitoring.

    Rules
    -----
    * CRITICAL status -> OPEN immediately.
    * Health score <= health_open_threshold -> OPEN.
    * Health score >= health_close_threshold -> CLOSED.
    * OPEN state after cooldown -> HALF_OPEN.
    """

    config = load_circuit_breaker_config()

    if not bool(
        config.get(
            "enabled",
            True,
        )
    ):

        return load_state()

    state = load_state()

    try:

        score = float(
            health_score
        )

    except Exception:

        score = None

    status = str(
        health_status
        if health_status is not None
        else "UNKNOWN"
    ).upper()

    open_threshold = float(
        config.get(
            "health_open_threshold",
            DEFAULT_CONFIG[
                "health_open_threshold"
            ],
        )
    )

    close_threshold = float(
        config.get(
            "health_close_threshold",
            DEFAULT_CONFIG[
                "health_close_threshold"
            ],
        )
    )

    failure_reason = reason or (
        f"System health is {status} "
        f"with score {score}."
    )

    # --------------------------------------------------------
    # CRITICAL STATUS
    # --------------------------------------------------------

    if status == "CRITICAL":

        return open_circuit(
            reason=failure_reason,
            health_score=score,
            health_status=status,
        )

    # --------------------------------------------------------
    # LOW HEALTH SCORE
    # --------------------------------------------------------

    if (
        score is not None
        and score <= open_threshold
    ):

        return open_circuit(
            reason=failure_reason,
            health_score=score,
            health_status=status,
        )

    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    if (
        score is not None
        and score >= close_threshold
        and status == "HEALTHY"
    ):

        return close_circuit(
            reason=(
                "Monitoring confirmed system recovery."
            ),
            health_score=score,
            health_status=status,
        )

    # --------------------------------------------------------
    # OPEN -> HALF_OPEN AFTER COOLDOWN
    # --------------------------------------------------------

    if state.get("state") == OPEN:

        cooldown_minutes = int(
            config.get(
                "cooldown_minutes",
                DEFAULT_CONFIG[
                    "cooldown_minutes"
                ],
            )
        )

        if cooldown_expired(
            state,
            cooldown_minutes,
        ):

            return half_open_circuit()

    # --------------------------------------------------------
    # SAVE CURRENT HEALTH
    # --------------------------------------------------------

    state["last_health_score"] = score

    state["last_health_status"] = status

    if reason:

        state["reason"] = reason

    save_state(state)

    return load_state()


# ============================================================
# PREDICTION PERMISSION
# ============================================================

def can_send_predictions() -> tuple[
    bool,
    str,
]:
    """
    Determine whether Telegram predictions may be sent.

    Returns
    -------
    tuple[bool, str]

        allowed
            True if predictions are allowed.

        reason
            Explanation of the current breaker state.
    """

    config = load_circuit_breaker_config()

    if not bool(
        config.get(
            "enabled",
            True,
        )
    ):

        return (
            True,
            "Circuit breaker is disabled.",
        )

    state = load_state()

    current_state = str(
        state.get(
            "state",
            CLOSED,
        )
    ).upper()

    # --------------------------------------------------------
    # CLOSED
    # --------------------------------------------------------

    if current_state == CLOSED:

        return (
            True,
            "Circuit breaker is CLOSED. "
            "Predictions are allowed.",
        )

    # --------------------------------------------------------
    # OPEN
    # --------------------------------------------------------

    if current_state == OPEN:

        cooldown_minutes = int(
            config.get(
                "cooldown_minutes",
                DEFAULT_CONFIG[
                    "cooldown_minutes"
                ],
            )
        )

        if cooldown_expired(
            state,
            cooldown_minutes,
        ):

            half_open_circuit()

            return (
                False,
                "Circuit breaker is HALF_OPEN. "
                "Waiting for a successful health check.",
            )

        return (
            False,
            "Circuit breaker is OPEN. "
            f"Predictions are blocked. "
            f"Reason: {state.get('reason')}",
        )

    # --------------------------------------------------------
    # HALF_OPEN
    # --------------------------------------------------------

    if current_state == HALF_OPEN:

        return (
            False,
            "Circuit breaker is HALF_OPEN. "
            "Predictions remain blocked until "
            "monitoring confirms recovery.",
        )

    # --------------------------------------------------------
    # UNKNOWN STATE
    # --------------------------------------------------------

    return (
        False,
        "Circuit breaker is in an unknown state. "
        "Predictions are blocked for safety.",
    )


# ============================================================
# STATUS
# ============================================================

def get_status() -> dict[str, Any]:
    """
    Return the current circuit breaker status.
    """

    config = load_circuit_breaker_config()

    state = load_state()

    allowed, message = (
        can_send_predictions()
    )

    return {
        "enabled": bool(
            config.get(
                "enabled",
                True,
            )
        ),
        "state": state.get(
            "state"
        ),
        "predictions_allowed": allowed,
        "message": message,
        "failure_count": state.get(
            "failure_count"
        ),
        "opened_at": state.get(
            "opened_at"
        ),
        "last_failure_at": state.get(
            "last_failure_at"
        ),
        "last_success_at": state.get(
            "last_success_at"
        ),
        "last_health_score": state.get(
            "last_health_score"
        ),
        "last_health_status": state.get(
            "last_health_status"
        ),
        "reason": state.get(
            "reason"
        ),
        "updated_at": state.get(
            "updated_at"
        ),
    }


# ============================================================
# MANUAL RESET
# ============================================================

def reset_circuit() -> dict[str, Any]:
    """
    Manually reset the circuit breaker.

    Useful for emergency recovery or manual operator control.
    """

    return close_circuit(
        reason=(
            "Manual circuit breaker reset."
        ),
        health_score=None,
        health_status="UNKNOWN",
    )


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """
    Command-line interface.

    Examples
    --------

    Show status:

        python src/circuit_breaker.py

    Open:

        python src/circuit_breaker.py open

    Close:

        python src/circuit_breaker.py close

    Reset:

        python src/circuit_breaker.py reset
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    import sys

    command = (
        sys.argv[1].strip().lower()
        if len(sys.argv) > 1
        else "status"
    )

    if command == "open":

        result = open_circuit(
            reason="Manual CLI open."
        )

    elif command == "close":

        result = close_circuit(
            reason="Manual CLI close."
        )

    elif command == "reset":

        result = reset_circuit()

    elif command == "status":

        result = get_status()

    else:

        print(
            "Unknown command."
        )

        print(
            "Usage: "
            "python src/circuit_breaker.py "
            "[status|open|close|reset]"
        )

        return 1

    print()

    print("=" * 60)

    print("CIRCUIT BREAKER STATUS")

    print("=" * 60)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
