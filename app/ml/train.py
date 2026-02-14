"""Model training script for sentiment analysis."""
import pandas as pd
import numpy as np
import logging
import joblib
import os
import sys
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
import warnings

from app.ml.features import FeatureEngineer

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SentimentModelTrainer:
    """Trainer for sentiment analysis model.

    Pipeline:
      1. Preprocess raw text
      2. TF-IDF vectorization (fitted on training set only)
      3. Extract 14 statistical features (polarity, subjectivity, …)
      4. Combine TF-IDF + scaled stats via hstack
      5. Train Logistic Regression on combined feature matrix
    """

    def __init__(
        self,
        data_path: str = "data/IMDB Dataset.csv",
        model_path: str = "models/sentiment_model.pkl",
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(max_features=10_000)

    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame | None:
        """Load dataset from CSV."""
        logger.info(f"Loading data from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            return None

    # ------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset."""
        logger.info("Preprocessing data...")
        df = df.dropna(subset=["review", "sentiment"]).copy()
        df["clean_review"] = df["review"].apply(
            self.feature_engineer.preprocess_text
        )
        df["label"] = (df["sentiment"] == "positive").astype(int)
        logger.info(
            f"Preprocessed {len(df)} samples. Label distribution:\n"
            f"{df['label'].value_counts().to_string()}"
        )
        return df

    # ------------------------------------------------------------------
    def _build_features(self, clean_texts, raw_texts, *, fit: bool):
        """Build combined TF-IDF + statistical feature matrix.

        Parameters
        ----------
        clean_texts : list[str]
            Preprocessed texts (for TF-IDF).
        raw_texts : list[str]
            Original texts (for statistical features — casing, punctuation
            must be preserved).
        fit : bool
            If True, fit the TF-IDF vectorizer and scaler on this data.
        """
        # TF-IDF
        if fit:
            self.feature_engineer.fit_tfidf(clean_texts)
        tfidf_matrix = self.feature_engineer.transform_tfidf(clean_texts)

        # 14 statistical features
        logger.info("Extracting statistical features...")
        stat_matrix = self.feature_engineer.extract_features_batch(raw_texts)

        if fit:
            stat_matrix = self.scaler.fit_transform(stat_matrix)
        else:
            stat_matrix = self.scaler.transform(stat_matrix)

        # Combine
        combined = hstack([tfidf_matrix, stat_matrix])
        logger.info(
            f"Combined feature matrix shape: {combined.shape} "
            f"(TF-IDF: {tfidf_matrix.shape[1]}, stats: {stat_matrix.shape[1]})"
        )
        return combined

    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame):
        """Train the sentiment model end-to-end."""
        logger.info("Starting model training...")

        df = self.preprocess(df)

        clean_texts = df["clean_review"].values.tolist()
        raw_texts = df["review"].values.tolist()
        y = df["label"].values

        # Split data (keep raw/clean aligned)
        (
            clean_train, clean_test,
            raw_train, raw_test,
            y_train, y_test,
        ) = train_test_split(
            clean_texts, raw_texts, y,
            test_size=0.2, random_state=42, stratify=y,
        )

        # Build combined features
        logger.info("Building training features...")
        X_train = self._build_features(clean_train, raw_train, fit=True)

        logger.info("Building test features...")
        X_test = self._build_features(clean_test, raw_test, fit=False)

        # Train logistic regression
        logger.info("Training Logistic Regression classifier...")
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Evaluate on test set
        self.evaluate(X_test, y_test)

        return self.model

    # ------------------------------------------------------------------
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if self.model is None:
            logger.error("Model not trained")
            return None

        predictions = self.model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions), 4),
            "recall": round(recall_score(y_test, predictions), 4),
            "f1": round(f1_score(y_test, predictions), 4),
        }
        logger.info(f"Model metrics: {metrics}")
        logger.info(
            "\nClassification Report:\n"
            + classification_report(
                y_test, predictions, target_names=["negative", "positive"]
            )
        )
        return metrics

    # ------------------------------------------------------------------
    def save_model(self):
        """Save trained model, vectorizer, scaler, and feature engineer."""
        if self.model is None:
            logger.error("No model to save")
            return False

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            artifact = {
                "model": self.model,
                "tfidf_vectorizer": self.feature_engineer.tfidf,
                "feature_engineer": self.feature_engineer,
                "scaler": self.scaler,
            }
            joblib.dump(artifact, self.model_path)
            logger.info(f"Model artifact saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False


if __name__ == "__main__":
    trainer = SentimentModelTrainer()
    df = trainer.load_data()
    if df is not None:
        trainer.train(df)
        trainer.save_model()
    else:
        logger.error("Training aborted: could not load data.")
        sys.exit(1)
