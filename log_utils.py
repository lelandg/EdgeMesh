"""Application logging with per-user storage and independent named loggers."""
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Retained for callers that used the previous module attribute.
the_logger = None


def _log_directory():
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "EdgeMesh" / "logs"


def setup_logger(name="edgemesh", log_file=None, level=logging.INFO,
                 format_string=None, date_format="%Y-%m-%d %H:%M"):
    """Configure a named logger once; an explicit log_file overrides user storage."""
    global the_logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(handler, "_edgemesh_handler", False) for handler in logger.handlers):
        path = Path(log_file) if log_file is not None else _log_directory() / "edgemesh.log"
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
