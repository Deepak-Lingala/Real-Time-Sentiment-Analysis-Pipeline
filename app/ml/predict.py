"""Prediction logic for sentiment analysis."""
import joblib
import logging
import os
from typing import Tuple

import numpy as np
from scipy.sparse import hstack

from app.ml.features import FeatureEngineer

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """Predictor for sentiment analysis using a trained sklearn pipeline.

    Loads the saved model artifact which contains:
      • Logistic Regression model
      • Fitted TF-IDF vectorizer
      • FeatureEngineer (for text preprocessing + statistical features)
      • StandardScaler (for normalizing statistical features)
    """

    LABELS = {0: "negative", 1: "positive"}

    def __init__(self, model_path: str = "models/sentiment_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_engineer: FeatureEngineer | None = None
        self.scaler = None
        self._loaded = False
        self._load_model()

    def _load_model(self) -> bool:
        """Load pre-trained model artifact from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}")
            return False

        try:
            artifact = joblib.load(self.model_path)
            self.model = artifact["model"]
            self.tfidf_vectorizer = artifact["tfidf_vectorizer"]
            self.feature_engineer = artifact["feature_engineer"]
            self.scaler = artifact["scaler"]
            self._loaded = True
            logger.info(f"Model artifact loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        """Check if the predictor is ready to make predictions."""
        return self._loaded and self.model is not None

    # ------------------------------------------------------------------
    def _build_features(self, raw_texts: list[str]):
        """Build combined TF-IDF + scaled statistical feature matrix."""
        clean_texts = [
            self.feature_engineer.preprocess_text(t) for t in raw_texts
        ]

        # TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.transform(clean_texts)

        # 14 statistical features (on *raw* text to preserve casing / punctuation)
        stat_matrix = self.feature_engineer.extract_features_batch(raw_texts)
        stat_matrix = self.scaler.transform(stat_matrix)

        return hstack([tfidf_matrix, stat_matrix])

    # ------------------------------------------------------------------
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for a given text.

        Args:
            text: Input text for sentiment analysis

        Returns:
            Tuple of (sentiment_label, confidence)
        """
        if not self.is_ready:
            logger.error("Model not loaded — cannot predict")
            return "unknown", 0.0

        try:
            features = self._build_features([text])

            probas = self.model.predict_proba(features)[0]
            predicted_class = int(np.argmax(probas))
            confidence = float(probas[predicted_class])

            sentiment = self.LABELS.get(predicted_class, "unknown")
            return sentiment, round(confidence, 4)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "unknown", 0.0

    def predict_batch(self, texts: list[str]) -> list[Tuple[str, float]]:
        """Predict sentiment for multiple texts (vectorized).

        Args:
            texts: List of input texts

        Returns:
            List of (sentiment, confidence) tuples
        """
        if not self.is_ready:
            logger.error("Model not loaded — cannot predict batch")
            return [("unknown", 0.0)] * len(texts)

        try:
            features = self._build_features(texts)

            probas = self.model.predict_proba(features)
            predicted_classes = np.argmax(probas, axis=1)

            results = []
            for idx, cls in enumerate(predicted_classes):
                sentiment = self.LABELS.get(int(cls), "unknown")
                confidence = round(float(probas[idx][cls]), 4)
                results.append((sentiment, confidence))

            return results

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return [("unknown", 0.0)] * len(texts)
