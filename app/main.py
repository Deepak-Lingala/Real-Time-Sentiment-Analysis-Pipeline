"""FastAPI application for real-time sentiment analysis."""
import hashlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
)
from app.ml.predict import SentimentPredictor
from app.cache import get_cached, set_cached
from app.database import create_tables, save_prediction, get_recent_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global predictor instance (initialized at startup)
# ---------------------------------------------------------------------------
predictor: SentimentPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    global predictor

    # Startup ---------------------------------------------------------------
    logger.info("Starting up — initializing predictor and database tables...")
    predictor = SentimentPredictor()
    if predictor.is_ready:
        logger.info("Sentiment model loaded successfully")
    else:
        logger.warning(
            "Sentiment model NOT available. Train the model first with: "
            "python -m app.ml.train"
        )

    # Create database tables (no-op if they already exist)
    create_tables()

    yield

    # Shutdown --------------------------------------------------------------
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentiment Analysis Pipeline",
    description="Real-time sentiment analysis API with caching and persistence",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _cache_key(text: str) -> str:
    """Generate a deterministic cache key for a text input."""
    return f"sentiment:{hashlib.sha256(text.encode()).hexdigest()}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Sentiment Analysis Pipeline API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_ready if predictor else False,
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment for a single text with caching and DB persistence."""
    if predictor is None or not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model first.",
        )

    # Check cache
    key = _cache_key(request.text)
    cached = get_cached(key)
    if cached:
        logger.info("Cache hit")
        return SentimentResponse(**cached)

    # Predict
    sentiment, confidence = predictor.predict(request.text)

    result = SentimentResponse(
        text=request.text, sentiment=sentiment, confidence=confidence
    )

    # Cache result
    set_cached(key, result.model_dump())

    # Persist to DB (fire-and-forget; failure doesn't block response)
    save_prediction(request.text, sentiment, confidence)

    return result


@app.post("/predict-batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchSentimentRequest):
    """Predict sentiment for multiple texts."""
    if predictor is None or not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model first.",
        )

    predictions = predictor.predict_batch(request.texts)

    results = []
    for text, (sentiment, confidence) in zip(request.texts, predictions):
        resp = SentimentResponse(
            text=text, sentiment=sentiment, confidence=confidence
        )
        results.append(resp)

        # Cache each result individually
        set_cached(_cache_key(text), resp.model_dump())

        # Persist each prediction
        save_prediction(text, sentiment, confidence)

    return BatchSentimentResponse(results=results)


@app.get("/history")
async def prediction_history(limit: int = 20):
    """Retrieve recent prediction history from the database."""
    records = get_recent_predictions(limit=limit)
    return {"count": len(records), "predictions": records}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
