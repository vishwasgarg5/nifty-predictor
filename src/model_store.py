"""Persistent storage for trained ML models."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


logger = logging.getLogger(__name__)


MODEL_STORE_VERSION = "1.0"

DEFAULT_MODEL_PATH = "data/models"


class ModelStore:
    """Save and load trained models for individual symbols."""

    def __init__(
        self,
        base_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:

        self.base_path = Path(
            base_path
        )

        self.base_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # PATH HELPERS
    # ========================================================

    def _safe_symbol(
        self,
        symbol: str,
    ) -> str:
        """Convert symbol into a safe directory name."""

        return (
            str(symbol)
            .strip()
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )

    def _symbol_path(
        self,
        symbol: str,
    ) -> Path:
        """Return directory path for a symbol."""

        safe_symbol = self._safe_symbol(
            symbol
        )

        return (
            self.base_path
            / safe_symbol
        )

    def _model_path(
        self,
        symbol: str,
    ) -> Path:
        """Return model file path."""

        return (
            self._symbol_path(
                symbol
            )
            / "pipeline.joblib"
        )

    def _metadata_path(
        self,
        symbol: str,
    ) -> Path:
        """Return metadata file path."""

        return (
            self._symbol_path(
                symbol
            )
            / "metadata.json"
        )

    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        symbol: str,
    ) -> bool:
        """Check whether a complete model exists."""

        return (
            self._model_path(
                symbol
            ).exists()
            and
            self._metadata_path(
                symbol
            ).exists()
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        symbol: str,
        pipeline: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save a trained model and metadata atomically.
        """

        if not symbol:

            raise ValueError(
                "symbol cannot be empty."
            )

        if pipeline is None:

            raise ValueError(
                "pipeline cannot be None."
            )

        symbol_dir = self._symbol_path(
            symbol
        )

        symbol_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = self._model_path(
            symbol
        )

        metadata_path = self._metadata_path(
            symbol
        )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        # ----------------------------------------------------
        # DEFAULT METADATA
        # ----------------------------------------------------

        payload: dict[str, Any] = {

            "store_version": (
                MODEL_STORE_VERSION
            ),

            "symbol": str(
                symbol
            ),

            "saved_at": now,

            "model_version": getattr(
                pipeline,
                "model_version",
                "unknown",
            ),

            "model_type": type(
                pipeline
            ).__name__,

            "training_rows": int(
                getattr(
                    pipeline,
                    "training_rows",
                    getattr(
                        pipeline,
                        "train_size",
                        0,
                    ),
                )
                or 0
            ),
        }

        # ----------------------------------------------------
        # MODEL METADATA
        # ----------------------------------------------------

        get_metadata = getattr(
            pipeline,
            "get_metadata",
            None,
        )

        if callable(
            get_metadata
        ):

            try:

                model_metadata = (
                    get_metadata()
                )

                if isinstance(
                    model_metadata,
                    dict,
                ):

                    payload.update(
                        model_metadata
                    )

            except Exception:

                logger.exception(
                    "Could not collect model metadata "
                    "for %s",
                    symbol,
                )

        # ----------------------------------------------------
        # USER METADATA
        # ----------------------------------------------------

        if metadata:

            payload.update(
                metadata
            )

        # Always ensure these values exist.

        payload["symbol"] = str(
            symbol
        )

        payload.setdefault(
            "saved_at",
            now,
        )

        payload.setdefault(
            "store_version",
            MODEL_STORE_VERSION,
        )

        model_tmp_path: Path | None = None

        metadata_tmp_path: Path | None = None

        try:

            # ------------------------------------------------
            # TEMP MODEL
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".joblib.tmp",
                dir=symbol_dir,
                delete=False,
            ) as temp_model:

                model_tmp_path = Path(
                    temp_model.name
                )

            # Write model.

            joblib.dump(
                pipeline,
                model_tmp_path,
            )

            # ------------------------------------------------
            # TEMP METADATA
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json.tmp",
                dir=symbol_dir,
                encoding="utf-8",
                delete=False,
            ) as temp_metadata:

                metadata_tmp_path = Path(
                    temp_metadata.name
                )

                json.dump(
                    payload,
                    temp_metadata,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )

            # ------------------------------------------------
            # ATOMIC REPLACE
            # ------------------------------------------------

            os.replace(
                model_tmp_path,
                model_path,
            )

            os.replace(
                metadata_tmp_path,
                metadata_path,
            )

            logger.info(
                "Saved model for %s to %s",
                symbol,
                model_path,
            )

        except Exception:

            logger.exception(
                "Failed to save model for %s",
                symbol,
            )

            raise

        finally:

            for temporary_path in (
                model_tmp_path,
                metadata_tmp_path,
            ):

                if (
                    temporary_path
                    and temporary_path.exists()
                ):

                    try:

                        temporary_path.unlink()

                    except OSError:

                        pass

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        symbol: str,
    ) -> tuple[
        Any,
        dict[str, Any],
    ] | None:
        """
        Load a saved model and metadata.

        Returns:

            (model, metadata)

        or:

            None
        """

        if not self.exists(
            symbol
        ):

            logger.debug(
                "No saved model found for %s",
                symbol,
            )

            return None

        model_path = self._model_path(
            symbol
        )

        metadata_path = self._metadata_path(
            symbol
        )

        try:

            # Load metadata first.

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            # Load model.

            pipeline = joblib.load(
                model_path
            )

            if pipeline is None:

                logger.error(
                    "Loaded model is None for %s",
                    symbol,
                )

                return None

            if not isinstance(
                metadata,
                dict,
            ):

                metadata = {}

            metadata.setdefault(
                "symbol",
                str(symbol),
            )

            metadata.setdefault(
                "model_source",
                "FALLBACK",
            )

            logger.info(
                "Loaded model for %s",
                symbol,
            )

            return (
                pipeline,
                metadata,
            )

        except Exception:

            logger.exception(
                "Failed to load model for %s",
                symbol,
            )

            return None

    # ========================================================
    # GET METADATA
    # ========================================================

    def get_metadata(
        self,
        symbol: str,
    ) -> dict[str, Any] | None:
        """Load metadata without loading the model."""

        metadata_path = self._metadata_path(
            symbol
        )

        if not metadata_path.exists():

            return None

        try:

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            if not isinstance(
                metadata,
                dict,
            ):

                return None

            return metadata

        except Exception:

            logger.exception(
                "Failed to read metadata for %s",
                symbol,
            )

            return None

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        symbol: str,
    ) -> bool:
        """Delete saved model files for a symbol."""

        model_path = self._model_path(
            symbol
        )

        metadata_path = self._metadata_path(
            symbol
        )

        deleted = False

        for path in (
            model_path,
            metadata_path,
        ):

            if path.exists():

                try:

                    path.unlink()

                    deleted = True

                except OSError:

                    logger.exception(
                        "Failed to delete %s",
                        path,
                    )

        # Try to remove empty symbol directory.

        symbol_dir = self._symbol_path(
            symbol
        )

        try:

            if symbol_dir.exists():

                symbol_dir.rmdir()

        except OSError:

            # Directory may contain
            # other files.
            pass

        if deleted:

            logger.info(
                "Deleted model for %s",
                symbol,
            )

        return deleted

    # ========================================================
    # LIST SYMBOLS
    # ========================================================

    def list_symbols(
        self,
    ) -> list[str]:
        """
        Return symbols with complete saved models.
        """

        if not self.base_path.exists():

            return []

        symbols: list[str] = []

        for path in self.base_path.iterdir():

            if not path.is_dir():

                continue

            model_path = (
                path
                / "pipeline.joblib"
            )

            metadata_path = (
                path
                / "metadata.json"
            )

            if (
                model_path.exists()
                and metadata_path.exists()
            ):

                symbols.append(
                    path.name
                )

        return sorted(
            symbols
        )


# ============================================================
# DEFAULT STORE
# ============================================================

_default_store = ModelStore()


# ============================================================
# COMPATIBILITY FUNCTIONS
#
# These functions are required because your existing
# prediction_pipeline.py searches for:
#
#     src.model_store.load_model()
#     src.model_store.get_model()
#     src.model_store.load()
# ============================================================

def save_model(
    symbol: str,
    model: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save a model using the default model store.
    """

    _default_store.save(
        symbol=symbol,
        pipeline=model,
        metadata=metadata,
    )


def load_model(
    symbol: str,
) -> tuple[
    Any,
    dict[str, Any],
] | None:
    """
    Load a model using the default model store.

    This is the primary compatibility function
    used by prediction_pipeline.py.
    """

    return _default_store.load(
        symbol
    )


def get_model(
    symbol: str,
) -> tuple[
    Any,
    dict[str, Any],
] | None:
    """Alias for load_model()."""

    return load_model(
        symbol
    )


def load(
    symbol: str,
) -> tuple[
    Any,
    dict[str, Any],
] | None:
    """Alias for load_model()."""

    return load_model(
        symbol
    )


def model_exists(
    symbol: str,
) -> bool:
    """Check whether a saved model exists."""

    return _default_store.exists(
        symbol
    )


def get_metadata(
    symbol: str,
) -> dict[str, Any] | None:
    """Get model metadata."""

    return _default_store.get_metadata(
        symbol
    )


def delete_model(
    symbol: str,
) -> bool:
    """Delete a saved model."""

    return _default_store.delete(
        symbol
    )


def list_models() -> list[str]:
    """Return all symbols with saved models."""

    return _default_store.list_symbols()
