"""
Logging utility module.

Configures structured logging for console output and optional file persistence
across all project modules.
"""

import logging
from pathlib import Path
import sys


def setup_logger(
    name: str = "",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures and returns a logger instance with formatted console and optional file handlers.

    Reuses existing handlers when possible and replaces an existing file handler when
    a new log path is requested.

    Args:
        name (str): Name of the logger instance. Defaults to "" (root logger).
        log_file (str | Path | None): Optional path to a file where logs will be saved.
            Defaults to None.
        level (int): Logging severity level (e.g., logging.INFO, logging.DEBUG).
            Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_console_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = file_path.resolve()
        file_handlers = [
            handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
        ]
        matching_handler = next(
            (handler for handler in file_handlers if Path(handler.baseFilename).resolve() == resolved_path),
            None,
        )

        if matching_handler is None:
            for handler in file_handlers:
                logger.removeHandler(handler)
                handler.close()

            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
