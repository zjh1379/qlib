import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.scheduling.router import get_manager


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_manager_is_initialized_after_lifespan(client):
    manager = get_manager()
    assert manager is not None
    assert manager._started is True
