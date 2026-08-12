from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


def _configure_test_environment(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_GENERATION_MODEL", "fake/generation")
    monkeypatch.setenv("NVIDIA_EMBEDDING_MODEL", "fake/embedding")
    monkeypatch.setenv("NVIDIA_EMBEDDING_DIMENSIONS", "4")


async def test_request_observability_adds_request_id_and_timing_headers(monkeypatch):
    _configure_test_environment(monkeypatch)
    from docintel.api.main import _request_observability_middleware

    app = FastAPI()
    app.middleware("http")(_request_observability_middleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ping", headers={"X-Request-ID": "req-test-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test-123"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


async def test_request_observability_generates_request_id_when_missing(monkeypatch):
    _configure_test_environment(monkeypatch)
    from docintel.api.main import _request_observability_middleware

    app = FastAPI()
    app.middleware("http")(_request_observability_middleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ping")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time-Ms"]) >= 0
