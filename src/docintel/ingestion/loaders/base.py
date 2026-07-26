"""
Loader plugin registry.

Each concrete loader (pdf_loader.py, docx_loader.py, ...) subclasses
`LoaderPlugin` and registers itself via `@register_loader(...)`. The
ingestion pipeline resolves the right loader for a given source purely
through this registry — it never imports a specific loader class by name,
so adding support for a new format never requires touching pipeline.py.

This module intentionally contains no format-specific logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

from docintel.core.logging import get_logger
from docintel.core.models import RawDocument

logger = get_logger(__name__)


class LoaderPlugin(ABC):
    """
    Base class for all loader plugins. Satisfies the `Loader` Protocol in
    core/interfaces.py — this ABC additionally gives us a registration
    hook, which a bare Protocol can't provide.
    """

    supported_extensions: tuple[str, ...] = ()
    """Lowercase extensions this loader handles, e.g. (".pdf",). Leave
    empty for URL-based loaders (e.g. the website loader)."""

    handles_urls: bool = False
    """True for loaders that operate on web URLs rather than local files."""

    def can_load(self, source: str | Path) -> bool:
        if self.handles_urls and isinstance(source, str):
            parsed = urlparse(source)
            if parsed.scheme in ("http", "https"):
                return True
        if isinstance(source, Path) or (isinstance(source, str) and not self.handles_urls):
            suffix = Path(str(source)).suffix.lower()
            return suffix in self.supported_extensions
        return False

    @abstractmethod
    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        """Load and return raw Document(s). Implemented by each concrete loader."""
        raise NotImplementedError


class LoaderRegistry:
    """
    Holds all registered loader plugins and resolves the correct one for
    a given source. A singleton instance (`registry`, below) is used
    throughout the app.
    """

    def __init__(self) -> None:
        self._loaders: list[LoaderPlugin] = []

    def register(self, loader: LoaderPlugin) -> None:
        logger.info(
            "loader_registered",
            loader=type(loader).__name__,
            extensions=loader.supported_extensions,
            handles_urls=loader.handles_urls,
        )
        self._loaders.append(loader)

    def resolve(self, source: str | Path) -> LoaderPlugin:
        for loader in self._loaders:
            if loader.can_load(source):
                return loader
        raise ValueError(
            f"No registered loader can handle source: {source!r}. "
            f"Registered loaders: {[type(l).__name__ for l in self._loaders]}"
        )

    def all(self) -> list[LoaderPlugin]:
        return list(self._loaders)


registry = LoaderRegistry()
"""Process-wide loader registry. Concrete loader modules call
`registry.register(SomeLoader())` at import time (see the
`register_loader` decorator below)."""


def register_loader(cls: type[LoaderPlugin]) -> type[LoaderPlugin]:
    """
    Class decorator that instantiates and registers a loader on import.

    Usage:
        @register_loader
        class PdfLoader(LoaderPlugin):
            supported_extensions = (".pdf",)
            async def load(self, source, knowledge_base_id): ...

    The loader is only registered once its module is actually imported —
    wiring which loader modules get imported happens in a later step
    (ingestion pipeline bootstrap), keeping this module free of a
    hardcoded list of formats.
    """
    registry.register(cls())
    return cls
