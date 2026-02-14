"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., min_length=1, max_length=5000)


class SentimentResponse(BaseModel):
    """Response model for sentiment prediction."""
    text: str
    sentiment: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment predictions."""
    results: list[SentimentResponse]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
