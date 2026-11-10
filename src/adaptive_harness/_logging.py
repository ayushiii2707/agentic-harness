"""Logging that cannot leak credentials.

All package loggers live under ``adaptive_harness``. A filter on that logger redacts secrets from every record,
including formatted arguments and exception text. Prompt content is never logged at INFO or above.
"""

from __future__ import annotations

import logging

from ._keys import redact

ROOT_LOGGER = "adaptive_harness"


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string must not break logging
            message = str(record.msg)
        record.msg = redact(message)
        record.args = None
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = (type(exc), exc, None)
        return True


_installed = False


def get_logger(name: str) -> logging.Logger:
    global _installed
    root = logging.getLogger(ROOT_LOGGER)
    if not _installed:
        root.addFilter(RedactingFilter())
        root.addHandler(logging.NullHandler())
        _installed = True
    logger = logging.getLogger(name if name.startswith(ROOT_LOGGER) else f"{ROOT_LOGGER}.{name}")
    if logger is not root and not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger
