"""Unit tests for the FeatureEngineer class."""
import pytest
import numpy as np
from app.ml.features import FeatureEngineer


@pytest.fixture
def engineer():
    return FeatureEngineer(min_df=1, max_df=1.0, max_features=100)


class TestPreprocessText:
    """Tests for text preprocessing."""

    def test_lowercases_text(self, engineer):
        assert engineer.preprocess_text("HELLO WORLD") == "hello world"

    def test_removes_urls(self, engineer):
        text = "Check out https://example.com for details"
        result = engineer.preprocess_text(text)
        assert "https" not in result
        assert "example" not in result

    def test_removes_email(self, engineer):
        result = engineer.preprocess_text("Contact user@example.com today")
        assert "@" not in result

    def test_removes_special_chars_and_digits(self, engineer):
        result = engineer.preprocess_text("Price is $100!! Great deal #1")
        assert "$" not in result
        assert "#" not in result
        assert "100" not in result

    def test_removes_extra_whitespace(self, engineer):
        result = engineer.preprocess_text("hello    world   foo")
        assert result == "hello world foo"

    def test_empty_string(self, engineer):
        assert engineer.preprocess_text("") == ""


class TestTfidf:
    """Tests for TF-IDF vectorizer."""

    def test_fit_and_transform(self, engineer):
        corpus = [
            "this movie was great",
            "terrible film awful acting",
            "wonderful experience loved it",
        ]
        engineer.fit_tfidf(corpus)
        result = engineer.transform_tfidf(["great movie"])
        assert result.shape[0] == 1
        assert result.shape[1] > 0

    def test_transform_before_fit_raises(self, engineer):
        with pytest.raises(ValueError, match="not fitted"):
            engineer.transform_tfidf(["hello world"])


class TestStatisticalFeatures:
    """Tests for the 14 statistical / linguistic features."""

    def test_returns_all_14_keys(self, engineer):
        features = engineer.extract_statistical_features("Hello world test!")
        assert set(features.keys()) == set(FeatureEngineer.STAT_FEATURE_NAMES)
        assert len(features) == 14

    def test_word_count(self, engineer):
        features = engineer.extract_statistical_features("one two three")
        assert features["word_count"] == 3

    def test_empty_string(self, engineer):
        features = engineer.extract_statistical_features("")
        assert features["length"] == 0
        assert features["word_count"] == 0
        assert features["upper_case_ratio"] == 0
        assert features["polarity"] == 0.0
        assert features["subjectivity"] == 0.0

    def test_polarity_positive(self, engineer):
        features = engineer.extract_statistical_features("This is an amazing, wonderful, great film!")
        assert features["polarity"] > 0

    def test_polarity_negative(self, engineer):
        features = engineer.extract_statistical_features("This is a terrible, awful, horrible movie.")
        assert features["polarity"] < 0

    def test_subjectivity_opinion(self, engineer):
        features = engineer.extract_statistical_features("I absolutely love this beautiful movie!")
        assert features["subjectivity"] > 0.3

    def test_punctuation_density(self, engineer):
        features = engineer.extract_statistical_features("Wow!!! Amazing!!!")
        assert features["punctuation_density"] > 0.2

    def test_exclamation_count(self, engineer):
        features = engineer.extract_statistical_features("Great! Awesome! Perfect!")
        assert features["exclamation_count"] == 3

    def test_question_count(self, engineer):
        features = engineer.extract_statistical_features("Really? Is it? Are you sure?")
        assert features["question_count"] == 3

    def test_negation_count(self, engineer):
        features = engineer.extract_statistical_features("I do not like this and I never will")
        assert features["negation_count"] >= 2

    def test_negation_contractions(self, engineer):
        features = engineer.extract_statistical_features("I don't think it isn't working")
        assert features["negation_count"] >= 2

    def test_unique_word_ratio(self, engineer):
        features = engineer.extract_statistical_features("the the the the")
        assert features["unique_word_ratio"] == 0.25

    def test_stopword_ratio(self, engineer):
        features = engineer.extract_statistical_features("the is a an")
        assert features["stopword_ratio"] > 0.5

    def test_digit_ratio(self, engineer):
        features = engineer.extract_statistical_features("test123")
        assert features["digit_ratio"] > 0.0


class TestBatchFeatures:
    """Tests for batch feature extraction."""

    def test_returns_correct_shape(self, engineer):
        texts = ["Hello world!", "Great movie.", "Bad film."]
        result = engineer.extract_features_batch(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 14)

    def test_single_text(self, engineer):
        result = engineer.extract_features_batch(["Single text here."])
        assert result.shape == (1, 14)
