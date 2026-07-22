"""Colored, component-tagged console logging shared across the kit.

Every line renders as::

    HH:MM:SS [tag] message

where ``[tag]`` is a short component label (``camera``, ``armleft``,
``armright``, ``ws``, ``webrtc``, ``session``, …) colored per component,
and the message is tinted by level (warnings yellow, errors red). Color
is emitted only when stderr is a TTY and ``NO_COLOR`` is unset, so piped
or redirected logs stay plain and grep-friendly.

Use it two ways::

    from vr_teleop_kit.log import setup_logging, get_logger
    setup_logging()                          # once, at startup
    log = get_logger(__name__, "armleft")    # tagged adapter
    log.info("ramping to rest")              #  -> 12:00:00 [armleft] ramping to rest
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import sys
from pathlib import Path

_RESET = "\033[0m"
_DIM = "\033[90m"

# Whole-message tint by level. INFO stays default-colored so normal output
# is not a wall of green.
_LEVEL_COLOR = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;37;41m",
}

# Fixed colors for well-known component tags; unknown tags get a stable
# pick from the palette (same tag -> same color every run).
_TAG_COLOR = {
    "camera": "\033[36m",    # cyan
    "webrtc": "\033[35m",    # magenta
    "ws": "\033[34m",        # blue
    "relay": "\033[34m",     # blue
    "armleft": "\033[32m",   # green
    "armright": "\033[95m",  # bright magenta
    "left": "\033[32m",      # aliases used by ramp_to_rest labels
    "right": "\033[95m",
    "session": "\033[96m",   # bright cyan
    "ik": "\033[36m",
}
_PALETTE = [
    "\033[36m", "\033[35m", "\033[34m", "\033[32m",
    "\033[33m", "\033[95m", "\033[96m", "\033[94m",
]


def _tag_color(tag: str) -> str:
    if tag in _TAG_COLOR:
        return _TAG_COLOR[tag]
    return _PALETTE[sum(map(ord, tag)) % len(_PALETTE)]


class ColorFormatter(logging.Formatter):
    """Formats records as ``HH:MM:SS [tag] message`` with optional color.

    The tag comes from ``record.tag`` (set via :func:`get_logger`) when
    present, else the last dotted segment of the logger name.
    """

    def __init__(self, color: bool = True) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", None) or record.name.rsplit(".", 1)[-1]
        ts = self.formatTime(record, self.datefmt)
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        if not self.color:
            return f"{ts} [{tag}] {msg}"
        lvl = _LEVEL_COLOR.get(record.levelno, "")
        body = f"{lvl}{msg}{_RESET}" if lvl else msg
        return f"{_DIM}{ts}{_RESET} {_tag_color(tag)}[{tag}]{_RESET} {body}"


def setup_logging(
    level: int = logging.INFO,
    force: bool = True,
    log_file: str | os.PathLike[str] | None = None,
    log_dir: str | os.PathLike[str] = "logs",
) -> Path | None:
    """Install the colored formatter on the root logger.

    ``force`` clears existing root handlers first (so calling this after a
    library's ``logging.basicConfig`` wins). Color auto-disables when
    stderr is not a TTY or ``NO_COLOR`` is set.

    Logs are also written to a plain-text file so runs are captured on disk.
    The destination is, in order of precedence: the ``log_file`` argument, the
    ``VR_TELEOP_LOG_FILE`` env var, or a timestamped file under ``log_dir``
    (``VR_TELEOP_LOG_DIR`` overrides the default ``logs/``). Set
    ``VR_TELEOP_NO_FILE_LOG`` to disable file logging entirely. Returns the
    path written to, or ``None`` when file logging is disabled.
    """
    color = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(color=color))
    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    if os.environ.get("VR_TELEOP_NO_FILE_LOG"):
        return None

    path = log_file or os.environ.get("VR_TELEOP_LOG_FILE") or None
    if path is None:
        base = Path(os.environ.get("VR_TELEOP_LOG_DIR") or log_dir)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = base / f"vr-teleop-{stamp}.log"
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:  # disk full, read-only fs, bad path — keep console logging
        root.warning("could not open log file %s: %s", path, exc)
        return None
    file_handler.setFormatter(ColorFormatter(color=False))
    root.addHandler(file_handler)
    root.info("logging to %s", path)
    return path


def get_logger(name: str, tag: str | None = None) -> logging.LoggerAdapter | logging.Logger:
    """Return a logger, optionally tagged so its lines print ``[tag]``."""
    lg = logging.getLogger(name)
    if tag:
        return logging.LoggerAdapter(lg, {"tag": tag})
    return lg
