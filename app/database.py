"""PostgreSQL database connection, ORM models, and helper functions."""
import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/sentiment_db",
)

try:
    engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info("Database engine initialized")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    engine = None
    SessionLocal = None
    Base = declarative_base()  # still define Base so ORM classes can be declared


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------
class PredictionRecord(Base):
    """SQLAlchemy model for storing prediction history."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<PredictionRecord(id={self.id}, sentiment={self.sentiment}, confidence={self.confidence})>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_tables():
    """Create all database tables if they don't exist."""
    if engine is None:
        logger.warning("Database engine not available — skipping table creation")
        return
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created / verified")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")


def get_db():
    """Yield a database session (FastAPI dependency)."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(text: str, sentiment: str, confidence: float):
    """Persist a single prediction to the database."""
    if SessionLocal is None:
        logger.warning("Database not available — prediction not saved")
        return None
    try:
        db = SessionLocal()
        record = PredictionRecord(
            text=text, sentiment=sentiment, confidence=confidence
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        db.close()
        return record
    except Exception as e:
        logger.warning(f"Failed to save prediction: {e}")
        return None


def get_recent_predictions(limit: int = 20) -> list[dict]:
    """Retrieve the most recent predictions from the database."""
    if SessionLocal is None:
        logger.warning("Database not available")
        return []
    try:
        db = SessionLocal()
        records = (
            db.query(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        db.close()
        return [
            {
                "id": r.id,
                "text": r.text,
                "sentiment": r.sentiment,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch predictions: {e}")
        return []
