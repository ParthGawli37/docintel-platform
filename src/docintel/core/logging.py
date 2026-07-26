"""
Structured logging configuration.

Call `configure_logging()` once at process startup (e.g. in api/main.py's
lifespan or a CLI entrypoint). Everywhere else, obtain a logger with
`get_logger(__name__)` and use structured key=value fields rather than
interpolated strings, so logs are machine-parseable in production.
"""

from __future__ import annotations

import logging
import sys

import structlog

from docintel.core.config import LogFormat, Settings


def configure_logging(settings: Settings) -> None:
    """Configure stdlib logging + structlog to agree on level/format/output."""

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level.value,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format is LogFormat.JSON:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.value)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    root_handler = logging.StreamHandler(sys.stdout)
    root_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [root_handler]
    root_logger.setLevel(settings.log_level.value)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to `name` (conventionally __name__)."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
