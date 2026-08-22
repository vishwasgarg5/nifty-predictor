#!/usr/bin/env python3

"""
Production Circuit Breaker.

Controls whether market predictions are allowed to be delivered
to Telegram.

State Machine
-------------

                    critical failure
        ┌─────────────────────────────────────┐
        │                                     ▼
     CLOSED ──────────────────────────────── OPEN
        ▲                                     │
        │                                     │ cooldown
        │                                     ▼
        │                                HALF_OPEN
        │                                     │
        │                    ┌────────────────┴───────────────┐
        │                    │                                │
        │                    ▼                                ▼
        └──────────── recovery success                     failure
                         CLOSED                            OPEN


The circuit breaker does NOT stop:

    - prediction generation
    - prediction ledger updates
    - actual outcome evaluation
    - model evaluation
    - drift detection

It only controls whether predictions may be delivered to Telegram.

Public API
----------

    can_send_predictions()
    get_status()
    register_failure()
    register_success()
    reset_circuit_breaker()
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("circuit_breaker")


# ============================================================
# TIME
# ============================================================

def utc_now() -> datetime:
    """Return current UTC datetime."""

    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    """Return current UTC timestamp."""

    return utc_now().isoformat()


def parse_datetime(
    value: Any,
) -> datetime | None:
    """Safely parse an ISO datetime."""

    if value is None:
        return None

    try:

        value = str(value)

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
# CONFIG
# ============================================================

def object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """Convert configuration object into dict."""

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


def load_config() -> Any:
    """Load project config."""

    try:

        from src.config import cfg

        return cfg

    except Exception as error:

        logger.warning(
            "Could not load config: %s",
            error,
        )

        return None


def get_circuit_breaker_config() -> dict[str, Any]:
    """
    Load circuit breaker configuration.

    Example:

        circuit_breaker:
            enabled: true
            failure_threshold: 2
            recovery_success_threshold: 2
            cooldown_minutes: 60
            critical_health_score: 40
    """

    defaults = {
        "enabled": True,
        "failure_threshold": 2,
        "recovery_success_threshold": 2,
        "cooldown_minutes": 60,
        "critical_health_score": 40.0,
    }

    cfg = load_config()

    if cfg is None:
        return defaults

    section = getattr(
        cfg,
        "circuit_breaker",
        None,
    )

    values = object_to_dict(
        section
    )

    result = defaults.copy()

    for key in defaults:

        if key not in values:
            continue

        value = values[key]

        if key == "enabled":

            result[key] = bool(value)

        elif key in {
            "failure_threshold",
            "recovery_success_threshold",
            "cooldown_minutes",
        }:

            try:
                result[key] = max(
                    1,
                    int(value),
                )
            except Exception:
                pass

        else:

            try:
                result[key] = float(value)
            except Exception:
                pass

    return result


# ============================================================
# STATE STORAGE
# ============================================================

def get_state_path() -> Path:
    """Return persistent state file path."""

    cfg = load_config()

    if cfg is not None:

        section = getattr(
            cfg,
            "circuit_breaker",
            None,
        )

        values = object_to_dict(
            section
        )

        state_file = values.get(
            "state_file"
        )

        if state_file:

            path = Path(
                str(state_file)
            )

            if not path.is_absolute():

                path = (
                    PROJECT_ROOT
                    / path
                )

            return path

    return (
        PROJECT_ROOT
        / "data"
        / "state"
        / "circuit_breaker.json"
    )


def default_state() -> dict[str, Any]:
    """Create default CLOSED state."""

    return {
        "state": "CLOSED",

        "failure_count": 0,
        "success_count": 0,

        "opened_at": None,
        "last_failure_at": None,
        "last_success_at": None,

        "last_reason": None,

        "health_score": 100.0,
        "health_status": "HEALTHY",

        "updated_at": utc_now_iso(),
    }


def load_state() -> dict[str, Any]:
    """Load persistent circuit breaker state."""

    path = get_state_path()

    if not path.exists():

        return default_state()

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            state = json.load(
                file
            )

        if not isinstance(
            state,
            dict,
        ):

            return default_state()

        result = default_state()

        result.update(
            state
        )

        return result

    except Exception as error:

        logger.error(
            "Could not load circuit breaker state: %s",
            error,
        )

        return default_state()


def save_state(
    state: dict[str, Any],
) -> None:
    """Persist circuit breaker state."""

    path = get_state_path()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state["updated_at"] = (
        utc_now_iso()
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=2,
            default=str,
        )


# ============================================================
# STATE TRANSITIONS
# ============================================================

def transition_to_open(
    state: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Move circuit breaker to OPEN."""

    state["state"] = "OPEN"

    state["opened_at"] = (
        utc_now_iso()
    )

    state["success_count"] = 0

    state["last_reason"] = reason

    logger.warning(
        "Circuit breaker OPEN | %s",
        reason,
    )

    return state


def transition_to_half_open(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Move circuit breaker to HALF_OPEN."""

    state["state"] = "HALF_OPEN"

    state["success_count"] = 0

    logger.warning(
        "Circuit breaker HALF_OPEN."
    )

    return state


def transition_to_closed(
    state: dict[str, Any],
    reason: str = "Recovered.",
) -> dict[str, Any]:
    """Move circuit breaker to CLOSED."""

    state["state"] = "CLOSED"

    state["failure_count"] = 0
    state["success_count"] = 0
    state["opened_at"] = None

    state["last_reason"] = reason

    logger.info(
        "Circuit breaker CLOSED | %s",
        reason,
    )

    return state


# ============================================================
# COOLDOWN
# ============================================================

def cooldown_expired(
    state: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """Check whether OPEN cooldown has expired."""

    opened_at = parse_datetime(
        state.get(
            "opened_at"
        )
    )

    if opened_at is None:
        return True

    cooldown = timedelta(
        minutes=config[
            "cooldown_minutes"
        ]
    )

    return utc_now() >= (
        opened_at
        + cooldown
    )


# ============================================================
# STATUS
# ============================================================

def get_status() -> dict[str, Any]:
    """
    Get current circuit breaker status.

    Automatically moves OPEN → HALF_OPEN when
    the cooldown period expires.
    """

    config = (
        get_circuit_breaker_config()
    )

    state = load_state()

    if not config.get(
        "enabled",
        True,
    ):

        state["state"] = "DISABLED"

        state[
            "predictions_allowed"
        ] = True

        state["reason"] = (
            "Circuit breaker disabled."
        )

        return state

    current_state = str(
        state.get(
            "state",
            "CLOSED",
        )
    ).upper()

    if (
        current_state == "OPEN"
        and cooldown_expired(
            state,
            config,
        )
    ):

        state = transition_to_half_open(
            state
        )

        save_state(
            state
        )

        current_state = "HALF_OPEN"

    allowed = current_state in {
        "CLOSED",
        "HALF_OPEN",
    }

    state[
        "predictions_allowed"
    ] = allowed

    if current_state == "OPEN":

        state["reason"] = (
            state.get(
                "last_reason"
            )
            or "Circuit breaker is OPEN."
        )

    elif current_state == "HALF_OPEN":

        state["reason"] = (
            "Recovery test in progress."
        )

    elif current_state == "CLOSED":

        state["reason"] = (
            state.get(
                "last_reason"
            )
            or "Circuit breaker is healthy."
        )

    return state


# ============================================================
# SEND DECISION
# ============================================================

def can_send_predictions() -> tuple[
    bool,
    str,
]:
    """
    Determine whether Telegram predictions may be sent.
    """

    status = get_status()

    allowed = bool(
        status.get(
            "predictions_allowed",
            False,
        )
    )

    reason = str(
        status.get(
            "reason",
            "Unknown circuit breaker state.",
        )
    )

    return (
        allowed,
        reason,
    )


# ============================================================
# FAILURE REGISTRATION
# ============================================================

def register_failure(
    reason: str,
    health_score: float | None = None,
    health_status: str | None = None,
) -> dict[str, Any]:
    """
    Register a production failure.

    Behaviour:

    CLOSED:
        failure count increases.
        Opens after failure_threshold.

    HALF_OPEN:
        any failure immediately returns to OPEN.

    OPEN:
        remains OPEN.
    """

    config = (
        get_circuit_breaker_config()
    )

    state = load_state()

    if not config.get(
        "enabled",
        True,
    ):

        return get_status()

    current_state = str(
        state.get(
            "state",
            "CLOSED",
        )
    ).upper()

    if health_score is not None:

        try:

            state["health_score"] = float(
                health_score
            )

        except Exception:
            pass

    if health_status is not None:

        state["health_status"] = str(
            health_status
        ).upper()

    state[
        "last_failure_at"
    ] = utc_now_iso()

    state[
        "last_reason"
    ] = str(reason)

    # HALF_OPEN failure immediately reopens.
    if current_state == "HALF_OPEN":

        state["failure_count"] = (
            int(
                state.get(
                    "failure_count",
                    0,
                )
            )
            + 1
        )

        state = transition_to_open(
            state,
            reason=(
                "HALF_OPEN recovery failed: "
                f"{reason}"
            ),
        )

        save_state(
            state
        )

        return get_status()

    # OPEN remains OPEN.
    if current_state == "OPEN":

        save_state(
            state
        )

        return get_status()

    # CLOSED failure.
    state["failure_count"] = (
        int(
            state.get(
                "failure_count",
                0,
            )
        )
        + 1
    )

    threshold = config[
        "failure_threshold"
    ]

    critical_health_score = config[
        "critical_health_score"
    ]

    immediate_open = False

    if health_score is not None:

        try:

            immediate_open = (
                float(
                    health_score
                )
                <= critical_health_score
            )

        except Exception:
            pass

    if (
        state["failure_count"]
        >= threshold
        or immediate_open
    ):

        state = transition_to_open(
            state,
            reason=reason,
        )

    save_state(
        state
    )

    return get_status()


# ============================================================
# SUCCESS REGISTRATION
# ============================================================

def register_success(
    health_score: float | None = None,
    health_status: str | None = None,
) -> dict[str, Any]:
    """
    Register successful production monitoring.

    CLOSED:
        resets failure count.

    HALF_OPEN:
        increments recovery success count.
        Closes after recovery_success_threshold.
    """

    config = (
        get_circuit_breaker_config()
    )

    state = load_state()

    if not config.get(
        "enabled",
        True,
    ):

        return get_status()

    current_state = str(
        state.get(
            "state",
            "CLOSED",
        )
    ).upper()

    if health_score is not None:

        try:

            state["health_score"] = float(
                health_score
            )

        except Exception:
            pass

    if health_status is not None:

        state["health_status"] = str(
            health_status
        ).upper()

    state[
        "last_success_at"
    ] = utc_now_iso()

    if current_state == "CLOSED":

        state["failure_count"] = 0
        state["success_count"] = 0

        state["last_reason"] = (
            "Successful health check."
        )

        save_state(
            state
        )

        return get_status()

    if current_state == "HALF_OPEN":

        state["success_count"] = (
            int(
                state.get(
                    "success_count",
                    0,
                )
            )
            + 1
        )

        threshold = config[
            "recovery_success_threshold"
        ]

        if (
            state["success_count"]
            >= threshold
        ):

            state = transition_to_closed(
                state,
                reason=(
                    "Recovered after "
                    f"{state['success_count']} "
                    "successful checks."
                ),
            )

        else:

            state["last_reason"] = (
                "HALF_OPEN recovery check "
                f"{state['success_count']}/"
                f"{threshold} successful."
            )

        save_state(
            state
        )

        return get_status()

    save_state(
        state
    )

    return get_status()


# ============================================================
# RESET
# ============================================================

def reset_circuit_breaker() -> dict[str, Any]:
    """
    Manually reset the circuit breaker to CLOSED.
    """

    state = default_state()

    state["last_reason"] = (
        "Manual reset."
    )

    save_state(
        state
    )

    return get_status()


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """Display circuit breaker status."""

    status = get_status()

    print()

    print("=" * 70)

    print("CIRCUIT BREAKER")

    print("=" * 70)

    print(
        f"State: "
        f"{status.get('state')}"
    )

    print(
        f"Predictions Allowed: "
        f"{status.get('predictions_allowed')}"
    )

    print(
        f"Failure Count: "
        f"{status.get('failure_count')}"
    )

    print(
        f"Success Count: "
        f"{status.get('success_count')}"
    )

    print(
        f"Health Score: "
        f"{status.get('health_score')}"
    )

    print(
        f"Health Status: "
        f"{status.get('health_status')}"
    )

    print(
        f"Reason: "
        f"{status.get('reason')}"
    )

    print()

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
