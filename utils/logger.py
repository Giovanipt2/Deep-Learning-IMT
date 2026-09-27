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

    Prevents handler duplication if the logger has already been initialized.

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

    # Avoid adding duplicate handlers if the logger is already configured
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream handler for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler for disk logging
    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
