from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from photo_sorter.schemas.config_models import LoggingConfig

_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(config: LoggingConfig) -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(config.level.upper())

    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=False,
        omit_repeated_times=False,
    )
    console.setLevel(config.level.upper())
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    if config.file_enabled:
        path = Path(config.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(config.level.upper())
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    return logging.getLogger("photo_sorter")
