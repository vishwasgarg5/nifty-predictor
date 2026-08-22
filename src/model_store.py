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


class ModelStore:
    """Save and load trained models for individual symbols."""

    def __init__(
        self,
        base_path: str | Path = "data/models",
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _symbol_path(
        self,
        symbol: str,
    ) -> Path:
        """Return safe directory path for a symbol."""

        safe_symbol = (
            str(symbol)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        return self.base_path / safe_symbol

    def _model_path(
        self,
        symbol: str,
    ) -> Path:
        return (
            self._symbol_path(symbol)
            / "pipeline.joblib"
        )

    def _metadata_path(
        self,
        symbol: str,
    ) -> Path:
        return (
            self._symbol_path(symbol)
            / "metadata.json"
        )

    def exists(
        self,
        symbol: str,
    ) -> bool:
        """Check whether a complete saved model exists."""

        return (
            self._model_path(symbol).exists()
            and self._metadata_path(symbol).exists()
        )

    def save(
        self,
        symbol: str,
        pipeline: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a trained pipeline and its metadata atomically."""

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

        payload = {
            "store_version": MODEL_STORE_VERSION,
            "symbol": symbol,
            "saved_at": now,
            "model_version": getattr(
                pipeline,
                "model_version",
                "unknown",
            ),
            "training_rows": int(
                getattr(
                    pipeline,
                    "training_rows",
                    0,
                )
                or 0
            ),
        }

        if metadata:
            payload.update(
                metadata
            )

        model_tmp_path: Path | None = None
        metadata_tmp_path: Path | None = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".joblib.tmp",
                dir=symbol_dir,
                delete=False,
            ) as temp_model:

                model_tmp_path = Path(
                    temp_model.name
                )

            joblib.dump(
                pipeline,
                model_tmp_path,
            )

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

    def load(
        self,
        symbol: str,
    ) -> tuple[Any, dict[str, Any]] | None:
        """Load a pipeline and metadata for a symbol."""

        if not self.exists(symbol):

            return None

        model_path = self._model_path(
            symbol
        )

        metadata_path = self._metadata_path(
            symbol
        )

        try:

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            pipeline = joblib.load(
                model_path
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

                return json.load(
                    file
                )

        except Exception:

            logger.exception(
                "Failed to read metadata for %s",
                symbol,
            )

            return None

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

        symbol_dir = self._symbol_path(
            symbol
        )

        try:

            if symbol_dir.exists():

                symbol_dir.rmdir()

        except OSError:

            # Directory may contain
            # additional files.
            pass

        return deleted

    def list_symbols(
        self,
    ) -> list[str]:
        """Return symbols with complete saved models."""

        if not self.base_path.exists():

            return []

        symbols: list[str] = []

        for path in self.base_path.iterdir():

            if not path.is_dir():

                continue

            model_path = (
                path / "pipeline.joblib"
            )

            metadata_path = (
                path / "metadata.json"
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
