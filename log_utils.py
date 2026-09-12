"""Application logging with per-user storage and independent named loggers."""
import logging
import hashlib
import re
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Retained for callers that used the previous module attribute.
the_logger = None


def _log_directory():
    from user_state import UserPaths
    return UserPaths.discover().logs_dir


def setup_logger(name="edgemesh", log_file=None, level=logging.INFO,
                 format_string=None, date_format="%Y-%m-%d %H:%M"):
    """Configure a named logger once; an explicit log_file overrides user storage."""
    global the_logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(handler, "_edgemesh_handler", False) for handler in logger.handlers):
        # Distinct rotating handlers must not race over the same default file.
        stem = "edgemesh" if name == "edgemesh" else (
            re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))[:60] + "-" + hashlib.sha256(str(name).encode()).hexdigest()[:10])
        path = Path(log_file) if log_file is not None else _log_directory() / f"{stem}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        handler._edgemesh_handler = True
        handler.setFormatter(logging.Formatter(
            format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt=date_format))
        logger.addHandler(handler)
    the_logger = logger
    return logger


def get_logger(name="edgemesh"):
    return setup_logger(name, level=logging.DEBUG)
