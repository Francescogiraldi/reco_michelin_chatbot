"""Logging utilities for Michelin Chatbot."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from ..config import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    enable_rich: bool = True
) -> None:
    """Setup application logging with loguru and rich.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_rich: Whether to enable rich formatting for console output
    """
    settings = get_settings()
    
    # Remove default logger
    logger.remove()
    
    # Set log level
    level = log_level or settings.log_level
    
    # Console handler with rich formatting
    if enable_rich:
        console = Console(stderr=True)
        logger.add(
            lambda msg: console.print(msg, end=""),
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=settings.log_format
        )
    
    # File handler
    if log_file:
        logger.add(
            log_file,
            level=level,
            format=settings.log_format,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
    else:
        # Default log file
        log_file = settings.logs_dir / "michelin_chatbot.log"
        logger.add(
            log_file,
            level=level,
            format=settings.log_format,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
    
    logger.info(f"Logging initialized with level: {level}")
    logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)