"""
Logging setup — call setup_logging() once at startup.
"""

import sys
from pathlib import Path

from loguru import logger

from config import settings


def setup_logging() -> None:
    logger.remove()     # remove default stderr handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
    )

    # Console
    logger.add(sys.stderr, format=log_format, level=settings.log_level, colorize=True)

    # File (rotating, 10 MB per file, keep 7 days)
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        format=log_format,
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        colorize=False,
        encoding="utf-8",
    )
