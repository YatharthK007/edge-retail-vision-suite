import logging
import os
import sys
from datetime import datetime

import config


# Module-level logger instance.
# Initialised to None so that any accidental call before setup_logger() raises a clear AttributeError rather than silently dropping messages.

_logger: logging.Logger | None = None



# Public API

def setup_logger() -> logging.Logger:
    """Initialise and return the ERVS application logger.

    Creates two handlers:
    - **FileHandler** — appends timestamped entries to config.LOG_FILE_PATH.
    - **StreamHandler** — mirrors every entry to stdout (console).

    The function is idempotent: calling it more than once returns the same
    logger without adding duplicate handlers.

    Returns
    -------
    logging.Logger
        The configured logger instance (also stored in the module-level
        ``_logger`` variable for use by the convenience functions below).
    """
    global _logger

    # Guard: if already initialised, return the existing logger immediately.
    if _logger is not None:
        return _logger

    # Ensure the directory that will hold the log file exists.
    log_dir = os.path.dirname(config.LOG_FILE_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Create (or retrieve) a named logger — using a fixed name means the same object is returned by logging.getLogger() anywhere in the project.
    logger = logging.getLogger("ERVS")
    logger.setLevel(logging.DEBUG)

    # Do not propagate to the root logger to avoid duplicate console output when third-party libraries (e.g. Ultralytics) configure the root logger.
    logger.propagate = False

    # Shared format: [2025-06-15 14:03:22] [INFO ] message
    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    #  File handler
    file_handler = logging.FileHandler(config.LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    #  Console handler 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    _logger = logger

    # Write a session separator so multiple runs are easy to distinguish in a long log file.
    _logger.info("=" * 60)
    _logger.info("ERVS session started  |  log → %s", config.LOG_FILE_PATH)
    _logger.info("=" * 60)

    return _logger


def _get_logger() -> logging.Logger:
    """Return the active logger, raising RuntimeError if not yet initialised.

    Internal helper used by every convenience function to guard against calls
    made before setup_logger() has been executed.

    Returns
    -------
    logging.Logger
        The active ERVS logger.

    Raises
    ------
    RuntimeError
        If setup_logger() has not been called yet.
    """
    if _logger is None:
        raise RuntimeError(
            "ERVS logger is not initialised. "
            "Call logger.setup_logger() before using any log_* functions."
        )
    return _logger



# Convenience logging functions

def log_entry(count: int) -> None:
    """Log a person-entry event detected by the tripwire.

    Parameters
    ----------
    count : int
        The updated total footfall count after this entry is registered.
    """
    _get_logger().info("EVENT=ENTRY  | footfall_count=%d", count)


def log_exit(count: int) -> None:
    """Log a person-exit event detected by the tripwire.

    Parameters
    ----------
    count : int
        The updated total footfall count after this exit is registered.
    """
    _get_logger().info("EVENT=EXIT   | footfall_count=%d", count)


def log_dwell(zone_name: str, duration_seconds: float) -> None:
    """Log a completed dwell-time event for a named interest zone.

    Called by utils.py when a tracked person leaves a DWELL_ZONE after
    exceeding config.DWELL_MIN_SECONDS.

    Parameters
    ----------
    zone_name : str
        The key from config.DWELL_ZONES (e.g. ``"billing_counter"``).
    duration_seconds : float
        How long (in seconds) the person was continuously inside the zone.
    """
    _get_logger().info(
        "EVENT=DWELL  | zone=%-20s | duration=%.2f s",
        zone_name,
        duration_seconds,
    )


def log_fps(fps: float, inference_ms: float) -> None:
    """Log a periodic performance-metrics snapshot.

    Intended to be called every 30 processed frames from main.py so the log
    file carries a lightweight performance trace without flooding it.

    Parameters
    ----------
    fps : float
        Frames processed per second at the moment of logging.
    inference_ms : float
        Model inference time in milliseconds for the most recent frame.
    """
    _get_logger().info(
        "PERF         | fps=%-6.1f | inference_ms=%.2f ms",
        fps,
        inference_ms,
    )


def log_warning(message: str) -> None:
    """Log a general warning message.

    Use for non-fatal issues such as a missing frame or a model returning no
    detections for an unexpected number of consecutive frames.

    Parameters
    ----------
    message : str
        Human-readable description of the warning condition.
    """
    _get_logger().warning("WARN         | %s", message)


def log_error(message: str) -> None:
    """Log an error message without raising an exception.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    """
    _get_logger().error("ERROR        | %s", message)


def log_session_end(total_entries: int, total_exits: int) -> None:
    """Log a summary banner when the video loop terminates cleanly.

    Parameters
    ----------
    total_entries : int
        Cumulative entry count for the entire session.
    total_exits : int
        Cumulative exit count for the entire session.
    """
    logger = _get_logger()
    logger.info("=" * 60)
    logger.info(
        "ERVS session ended   | total_entries=%d | total_exits=%d",
        total_entries,
        total_exits,
    )
    logger.info("=" * 60)