"""Logging configuration for conference agent."""

import logging
import sys


def setup_logger(name: str = __name__, level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Set up logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers from this logger
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """Get or create logger with name."""
    return logging.getLogger(name)
