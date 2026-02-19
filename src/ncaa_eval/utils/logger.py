"""Structured logging with configurable verbosity levels.

Provides project-wide logging configuration using Python's standard
`logging` module. Four verbosity levels map to standard (and one custom)
Python log levels:

    ========  ==============  =====
    Project   Python level    Value
    ========  ==============  =====
    QUIET     WARNING          30
    NORMAL    INFO             20
    VERBOSE   VERBOSE (custom) 15
    DEBUG     DEBUG            10
    ========  ==============  =====

Usage:
    Configure once at application startup, then obtain named loggers
    anywhere in the codebase::

        >>> from ncaa_eval.utils.logger import configure_logging, get_logger
        >>> configure_logging("VERBOSE")
        >>> log = get_logger("ingest")
        >>> log.info("Loading data...")

    The verbosity can also be controlled via the `NCAA_EVAL_LOG_LEVEL`
    environment variable (case-insensitive).  An explicit `level` argument
    to `configure_logging` takes precedence over the environment variable,
    which in turn takes precedence over the default (`NORMAL`).
"""

from __future__ import annotations

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Custom VERBOSE level (between INFO=20 and DEBUG=10)
# ---------------------------------------------------------------------------

VERBOSE: int = 15
"""Custom log level between INFO and DEBUG for detailed operational output."""

logging.addLevelName(VERBOSE, "VERBOSE")

# ---------------------------------------------------------------------------
# Convenience aliases for all project verbosity levels
# ---------------------------------------------------------------------------

QUIET: int = logging.WARNING
"""Project verbosity that suppresses routine output (maps to WARNING=30)."""

NORMAL: int = logging.INFO
"""Default project verbosity (maps to INFO=20)."""

DEBUG: int = logging.DEBUG
"""Full diagnostic output (maps to DEBUG=10)."""

# ---------------------------------------------------------------------------
# Internal mapping from project level names to numeric values
# ---------------------------------------------------------------------------

_LEVEL_MAP: dict[str, int] = {
    "QUIET": QUIET,
    "NORMAL": NORMAL,
    "VERBOSE": VERBOSE,
    "DEBUG": DEBUG,
}

_ROOT_LOGGER_NAME: str = "ncaa_eval"
_LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure project-wide logging with the given verbosity level.

    Resolution order:

    1. Explicit `level` argument (if not ``None``).
    2. ``NCAA_EVAL_LOG_LEVEL`` environment variable.
    3. ``"NORMAL"`` default.

    Args:
        level: One of ``"QUIET"``, ``"NORMAL"``, ``"VERBOSE"``, or
            ``"DEBUG"`` (case-insensitive).  ``None`` means fall through
            to the environment variable or default.

    Raises:
        ValueError: If the resolved level name is not recognised.

    Example:
        >>> from ncaa_eval.utils.logger import configure_logging, get_logger
        >>> configure_logging("VERBOSE")
        >>> log = get_logger("ingest")
        >>> log.info("Loading data...")
    """
    resolved: str = level if level is not None else os.environ.get("NCAA_EVAL_LOG_LEVEL", "NORMAL")
    resolved_upper = resolved.upper()

    if resolved_upper not in _LEVEL_MAP:
        msg = f"Unknown log level {resolved!r}. Valid levels: {', '.join(sorted(_LEVEL_MAP))}"
        raise ValueError(msg)

    numeric_level = _LEVEL_MAP[resolved_upper]

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(numeric_level)

    # Remove existing handlers to prevent duplicates on re-call.
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)

    # Prevent double-logging via the Python root logger.
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``ncaa_eval`` hierarchy.

    Args:
        name: Dot-separated path appended to the root ``ncaa_eval`` logger
            (e.g. ``"transform.features"`` yields ``ncaa_eval.transform.features``).

    Returns:
        A `logging.Logger` instance.

    Example:
        >>> log = get_logger("transform.features")
        >>> log.info("Computing features for season %d", 2025)
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
