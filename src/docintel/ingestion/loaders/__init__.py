"""
Loader bootstrap.

Importing a loader module triggers its `@register_loader` decorator,
registering it into the global registry (see base.py). This module is
the single, explicit place that lists which loader modules exist —
the pipeline orchestrator (pipeline.py, a later step) calls
`bootstrap_loaders()` once at startup and never needs its own list.

Adding a new format = write a new loader module + add one import line
here. Nothing else in the codebase changes.
"""

from __future__ import annotations

_bootstrapped = False


def bootstrap_loaders() -> None:
    """Import all loader modules exactly once, registering them."""
    global _bootstrapped
    if _bootstrapped:
        return

    from docintel.ingestion.loaders import (  # noqa: F401
        html_loader,
        image_ocr_loader,
        office_loaders,
        text_loaders,
        web_loader,
    )

    _bootstrapped = True
