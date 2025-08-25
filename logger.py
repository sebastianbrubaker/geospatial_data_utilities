# Author: Sebastian Brubaker

import logging
import os

def setup_logger(name: str = "logger", log_file: str = "logs/run.log", level=logging.INFO):
    """
    Reusable logger setup function.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Attach handlers (avoid duplicates)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger