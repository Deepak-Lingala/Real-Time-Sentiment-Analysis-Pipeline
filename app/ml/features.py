"""Feature engineering for sentiment analysis."""
import numpy as np
import logging
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob

logger = logging.getLogger(__name__)

# Negation words for negation pattern detection
NEGATION_WORDS = frozenset([
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "without", "hardly", "barely",
    "scarcely", "rarely", "seldom",
])

# Common English stopwords (lightweight set to avoid NLTK dependency at runtime)
STOPWORDS = frozenset([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now",
])


class FeatureEngineer:
    """Feature engineering for text data.

    Provides:
      • Text preprocessing (lowering, URL/email/special char removal)
      • TF-IDF vectorization
      • 14 statistical / linguistic features per text
    """

    # Feature names in fixed order (for reproducibility)
    STAT_FEATURE_NAMES = [
        "length",
        "word_count",
        "avg_word_length",
        "sentence_count",
        "upper_case_ratio",
        "polarity",
        "subjectivity",
        "punctuation_density",
        "exclamation_count",
        "question_count",
        "negation_count",
        "unique_word_ratio",
        "stopword_ratio",
        "digit_ratio",
    ]

    def __init__(self, min_df: int = 2, max_df: float = 0.95, max_features: int = 5000):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vectorizer = None
        self.tfidf = None

    # ------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ------------------------------------------------------------------
    # TF-IDF vectorization
    # ------------------------------------------------------------------
    def fit_tfidf(self, texts: list[str]):
        """Fit TF-IDF vectorizer on texts."""
        logger.info("Fitting TF-IDF vectorizer...")
        self.tfidf = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            stop_words='english',
        )
        self.tfidf.fit(texts)
        logger.info(f"TF-IDF vectorizer fitted. Vocabulary size: {len(self.tfidf.vocabulary_)}")
        return self.tfidf

    def transform_tfidf(self, texts: list[str]):
        """Transform texts using fitted TF-IDF vectorizer."""
        if self.tfidf is None:
            raise ValueError("TF-IDF vectorizer not fitted")
        return self.tfidf.transform(texts)

    # ------------------------------------------------------------------
    # Count vectorizer (kept for optional use)
    # ------------------------------------------------------------------
    def fit_count_vectorizer(self, texts: list[str]):
        """Fit Count vectorizer on texts."""
        logger.info("Fitting Count vectorizer...")
        self.vectorizer = CountVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            stop_words='english',
        )
        self.vectorizer.fit(texts)
        logger.info(f"Count vectorizer fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self.vectorizer

    # ------------------------------------------------------------------
    # Statistical / linguistic features (14 features)
    # ------------------------------------------------------------------
    def extract_statistical_features(self, text: str) -> dict:
        """Extract 14 statistical and linguistic features from *raw* text.

        Uses the **original (unprocessed)** text so that punctuation,
        casing, and digit information are preserved.
        """
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # TextBlob polarity & subjectivity
        blob = TextBlob(text)

        # Punctuation density
        punct_count = sum(1 for c in text if c in string.punctuation)

        # Negation: check raw words + contractions like "n't"
        lower_words = [w.lower().strip(string.punctuation) for w in words]
        negation_count = sum(
            1 for w in lower_words if w in NEGATION_WORDS
        ) + sum(
            1 for w in words if "n't" in w.lower()
        )

        # Stopword ratio
        stopword_hits = sum(1 for w in lower_words if w in STOPWORDS)

        features = {
            "length": char_count,
            "word_count": word_count,
            "avg_word_length": (
                float(np.mean([len(w) for w in words])) if word_count > 0 else 0.0
            ),
            "sentence_count": max(len(re.split(r'[.!?]+', text)), 1),
            "upper_case_ratio": (
                sum(1 for c in text if c.isupper()) / char_count
                if char_count > 0 else 0.0
            ),
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "punctuation_density": (
                punct_count / char_count if char_count > 0 else 0.0
            ),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "negation_count": negation_count,
            "unique_word_ratio": (
                len(set(lower_words)) / word_count if word_count > 0 else 0.0
            ),
            "stopword_ratio": (
                stopword_hits / word_count if word_count > 0 else 0.0
            ),
            "digit_ratio": (
                sum(1 for c in text if c.isdigit()) / char_count
                if char_count > 0 else 0.0
            ),
        }
        return features

    def extract_features_batch(self, texts: list[str]) -> np.ndarray:
        """Extract statistical features for a list of texts.

        Returns a (n_samples, 14) numpy array with features in
        ``STAT_FEATURE_NAMES`` order.
        """
        rows = []
        for text in texts:
            feat = self.extract_statistical_features(text)
            rows.append([feat[name] for name in self.STAT_FEATURE_NAMES])
        return np.array(rows, dtype=np.float64)
