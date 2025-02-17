# faux_lingo/utils/logging.py

"""
Logging configuration using loguru.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO", log_file: str | Path | None = None, rotation: str = "20 MB"
) -> None:
    """
    Configure loguru logger with console and optional file outputs.

    Args:
        level: Minimum log level to display
        log_file: Optional path to log file
        rotation: When to rotate log file (size or time interval)
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=level,
    )

    # Add file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            compression="zip",
        )

    logger.info("Logging configured at level {}", level)
