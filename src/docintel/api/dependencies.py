"""FastAPI dependency accessors -- pull the AppContainer off app.state."""

from __future__ import annotations

from fastapi import Request

from docintel.api.container import AppContainer


def get_container(request: Request) -> AppContainer:
    container: AppContainer = request.app.state.container
    return container
