"""Integration tests for the FastAPI application."""
import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def mock_predictor():
    """Mock the global predictor so tests don't need a trained model."""
    mock = MagicMock()
    mock.is_ready = True
    mock.predict.return_value = ("positive", 0.95)
    mock.predict_batch.return_value = [
        ("positive", 0.95),
        ("negative", 0.87),
    ]
    return mock


@pytest.mark.asyncio
async def test_root():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_predict(mock_predictor):
    with patch("app.main.predictor", mock_predictor), \
         patch("app.main.get_cached", return_value=None), \
         patch("app.main.set_cached"), \
         patch("app.main.save_prediction"):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/predict",
                json={"text": "This movie was amazing!"},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] == 0.95
    assert data["text"] == "This movie was amazing!"


@pytest.mark.asyncio
async def test_predict_batch(mock_predictor):
    with patch("app.main.predictor", mock_predictor), \
         patch("app.main.get_cached", return_value=None), \
         patch("app.main.set_cached"), \
         patch("app.main.save_prediction"):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/predict-batch",
                json={"texts": ["Great movie!", "Terrible film."]},
            )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["sentiment"] == "positive"
    assert data["results"][1]["sentiment"] == "negative"


@pytest.mark.asyncio
async def test_predict_model_not_loaded():
    """When predictor is None or not ready, should return 503."""
    with patch("app.main.predictor", None):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/predict",
                json={"text": "test"},
            )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_predict_empty_text():
    """Empty text should be rejected by Pydantic validation (min_length=1)."""
    with patch("app.main.predictor", MagicMock(is_ready=True)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/predict",
                json={"text": ""},
            )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_history():
    with patch("app.main.get_recent_predictions", return_value=[]):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "count" in data
