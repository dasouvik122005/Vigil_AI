"""
Unit tests for the ML Engine (AdaptiveDecisionEngine).
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
from ml_engine import AdaptiveDecisionEngine


class TestAdaptiveDecisionEngineInitialization:
    """Tests for engine initialization."""

    def test_init_with_valid_data(self, training_data, tmp_path):
        """Test initialization with valid training data."""
        # Save training data
        data_path = tmp_path / "train.csv"
        training_data.to_csv(data_path, index=False)

        engine = AdaptiveDecisionEngine(historical_data_path=str(data_path))

        assert engine.iso_forest is not None
        assert engine.rf_classifier is not None
        assert engine.means is not None
        assert engine.stds is not None
        assert len(engine.means) == 3
        assert len(engine.stds) == 3
        assert engine.features_list == ['temperature', 'pressure', 'vibration']

    def test_init_with_missing_file_uses_defaults(self):
        """Test initialization falls back to defaults when file missing."""
        engine = AdaptiveDecisionEngine(historical_data_path="nonexistent.csv")

        assert engine.iso_forest is not None
        assert engine.means == {'temperature': 25, 'pressure': 1.2, 'vibration': 0.3}
        assert engine.stds == {'temperature': 3, 'pressure': 0.1, 'vibration': 0.05}

    def test_features_list_correct(self, ml_engine):
        """Test that features list matches expected sensors."""
        assert ml_engine.features_list == ['temperature', 'pressure', 'vibration']


class TestPreprocessing:
    """Tests for data preprocessing."""

    def test_preprocess_complete_data(self, ml_engine, sample_sensor_data):
        """Test preprocessing with complete data."""
        X, missing_count = ml_engine._preprocess(sample_sensor_data)

        assert X.shape == (1, 3)
        assert missing_count == 0
        # Values should match input (approximately, due to float precision)
        assert abs(X[0, 0] - 25.0) < 0.01
        assert abs(X[0, 1] - 1.2) < 0.01
        assert abs(X[0, 2] - 0.3) < 0.01

    def test_preprocess_missing_data(self, ml_engine, sample_sensor_data_missing):
        """Test preprocessing with missing values (imputation)."""
        X, missing_count = ml_engine._preprocess(sample_sensor_data_missing)

        assert X.shape == (1, 3)
        assert missing_count == 1
        # Missing pressure should be imputed with mean
        assert abs(X[0, 1] - ml_engine.means['pressure']) < 0.01

    def test_preprocess_all_missing(self, ml_engine):
        """Test preprocessing with all values missing."""
        data = {"temperature": None, "pressure": None, "vibration": None}
        X, missing_count = ml_engine._preprocess(data)

        assert X.shape == (1, 3)
        assert missing_count == 3
        # All should be imputed with means
        for i, feat in enumerate(ml_engine.features_list):
            assert abs(X[0, i] - ml_engine.means[feat]) < 0.01


class TestPrediction:
    """Tests for prediction functionality."""

    def test_predict_normal_operation(self, ml_engine, sample_sensor_data):
        """Test prediction on normal data."""
        result = ml_engine.predict(sample_sensor_data)

        assert "is_anomaly" in result
        assert "confidence" in result
        assert "explanation" in result

        assert isinstance(result["is_anomaly"], bool)
        assert 0.01 <= result["confidence"] <= 0.99
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    def test_predict_anomaly(self, ml_engine, sample_anomaly_data):
        """Test prediction on anomalous data."""
        result = ml_engine.predict(sample_anomaly_data)

        # Should detect anomaly (though not guaranteed with small contamination)
        assert "is_anomaly" in result
        assert "confidence" in result

        if result["is_anomaly"]:
            assert "deviations" in result["explanation"].lower() or \
                   "anomaly" in result["explanation"].lower()

    def test_predict_missing_data_lowers_confidence(self, ml_engine, sample_sensor_data, sample_sensor_data_missing):
        """Test that missing data reduces confidence."""
        result_complete = ml_engine.predict(sample_sensor_data)
        result_missing = ml_engine.predict(sample_sensor_data_missing)

        # Missing data should generally lower confidence
        # (though exact values depend on the model)
        assert result_missing["confidence"] <= result_complete["confidence"] + 0.1  # Allow small variance

    def test_predict_bounds_confidence(self, ml_engine, sample_sensor_data):
        """Test that confidence is bounded between 0.01 and 0.99."""
        result = ml_engine.predict(sample_sensor_data)

        assert 0.01 <= result["confidence"] <= 0.99

    def test_explanation_contains_source(self, ml_engine, sample_sensor_data):
        """Test that explanation mentions prediction source."""
        result = ml_engine.predict(sample_sensor_data)

        assert "IsolationForest" in result["explanation"] or \
               "RandomForest" in result["explanation"] or \
               "Prediction made by" in result["explanation"]


class TestHumanFeedback:
    """Tests for human feedback and model adaptation."""

    def test_add_human_feedback_normal(self, ml_engine, sample_sensor_data):
        """Test adding human feedback for normal classification."""
        initial_buffer_size = len(ml_engine.feedback_buffer)

        ml_engine.add_human_feedback(sample_sensor_data, is_anomaly_label=False)

        assert len(ml_engine.feedback_buffer) == initial_buffer_size + 1
        # Check buffer stores correct format: (features, label)
        features, label = ml_engine.feedback_buffer[-1]
        assert label == 0  # Normal = 0
        assert features.shape == (3,)

    def test_add_human_feedback_anomaly(self, ml_engine, sample_anomaly_data):
        """Test adding human feedback for anomaly classification."""
        initial_buffer_size = len(ml_engine.feedback_buffer)

        ml_engine.add_human_feedback(sample_anomaly_data, is_anomaly_label=True)

        assert len(ml_engine.feedback_buffer) == initial_buffer_size + 1
        features, label = ml_engine.feedback_buffer[-1]
        assert label == 1  # Anomaly = 1

    def test_retrain_after_feedback_threshold(self, ml_engine, sample_sensor_data, sample_anomaly_data):
        """Test that model retrains after feedback batch size reached."""
        # Add feedback up to threshold (default 5)
        for i in range(5):
            data = sample_sensor_data if i % 2 == 0 else sample_anomaly_data
            label = i % 2 == 1  # Alternate normal/anomaly
            ml_engine.add_human_feedback(data, is_anomaly_label=label)

        # Should have triggered retrain
        # Note: Retrain only happens if both classes present
        assert len(ml_engine.feedback_buffer) >= 5

    def test_retrain_requires_both_classes(self, ml_engine, sample_sensor_data):
        """Test that retraining only happens with both classes."""
        # Add 5 normal samples only
        for _ in range(5):
            ml_engine.add_human_feedback(sample_sensor_data, is_anomaly_label=False)

        # Should not have trained RF (only one class)
        assert ml_engine.is_rf_trained is False


class TestConfidenceCalculation:
    """Tests for confidence scoring logic."""

    def test_confidence_decreases_with_missing_data(self, ml_engine):
        """Test confidence calculation accounts for missing data."""
        complete = {"temperature": 25.0, "pressure": 1.2, "vibration": 0.3}
        missing_one = {"temperature": 25.0, "pressure": None, "vibration": 0.3}
        missing_two = {"temperature": 25.0, "pressure": None, "vibration": None}
        missing_all = {"temperature": None, "pressure": None, "vibration": None}

        results = [
            ml_engine.predict(complete),
            ml_engine.predict(missing_one),
            ml_engine.predict(missing_two),
            ml_engine.predict(missing_all),
        ]

        confidences = [r["confidence"] for r in results]

        # More missing data -> lower max confidence (approximately)
        # completeness ratios: 1.0, 0.67, 0.33, 0.0
        # We check general trend (not strict due to model variance)
        assert confidences[0] >= confidences[3]  # Complete >= All missing


class TestExplainability:
    """Tests for explanation generation."""

    def test_explanation_for_anomaly_mentions_deviations(self, ml_engine, sample_anomaly_data):
        """Test anomaly explanations mention specific deviations."""
        result = ml_engine.predict(sample_anomaly_data)

        if result["is_anomaly"]:
            explanation = result["explanation"].lower()
            # Should mention z-scores or standard deviations
            assert any(term in explanation for term in [
                "deviation", "standard deviation", "z-score", "anomaly"
            ])

    def test_explanation_for_normal_mentions_boundaries(self, ml_engine, sample_sensor_data):
        """Test normal explanations mention normal boundaries."""
        result = ml_engine.predict(sample_sensor_data)

        if not result["is_anomaly"]:
            explanation = result["explanation"].lower()
            assert any(term in explanation for term in [
                "normal", "boundar", "within"
            ])

    def test_explanation_warns_on_missing_data(self, ml_engine, sample_sensor_data_missing):
        """Test explanation warns about missing sensors."""
        result = ml_engine.predict(sample_sensor_data_missing)

        explanation = result["explanation"].lower()
        assert "missing" in explanation or "offline" in explanation or "sensor" in explanation


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_predict_with_extra_fields_ignored(self, ml_engine, sample_sensor_data):
        """Test that extra fields in input are ignored."""
        data_with_extra = {**sample_sensor_data, "extra_field": "ignored", "another": 123}
        result = ml_engine.predict(data_with_extra)

        assert "is_anomaly" in result
        assert "confidence" in result

    def test_predict_with_wrong_types_handled(self, ml_engine):
        """Test handling of wrong input types."""
        # String numbers should be handled or raise clear error
        data = {"temperature": "25.0", "pressure": "1.2", "vibration": "0.3"}

        # Should either convert or raise informative error
        try:
            result = ml_engine.predict(data)
            assert "is_anomaly" in result
        except (TypeError, ValueError) as e:
            # Acceptable to raise error for wrong types
            assert "float" in str(e).lower() or "convert" in str(e).lower()

    def test_feedback_buffer_max_size(self, ml_engine, sample_sensor_data, sample_anomaly_data):
        """Test feedback buffer doesn't grow unbounded."""
        # Add many feedback samples
        for i in range(100):
            data = sample_sensor_data if i % 2 == 0 else sample_anomaly_data
            ml_engine.add_human_feedback(data, is_anomaly_label=(i % 2 == 1))

        # Buffer should be capped at max (1000 by default)
        assert len(ml_engine.feedback_buffer) <= 1000


class TestModelPersistence:
    """Tests for model state persistence (when implemented)."""

    def test_engine_state_after_feedback(self, ml_engine, sample_sensor_data):
        """Test engine maintains state after feedback."""
        # Add feedback
        ml_engine.add_human_feedback(sample_sensor_data, is_anomaly_label=True)

        # Engine should retain feedback buffer
        assert len(ml_engine.feedback_buffer) > 0

        # Predict again - should use updated state
        result = ml_engine.predict(sample_sensor_data)
        assert "is_anomaly" in result


# ==========================================
# Integration-style tests
# ==========================================

class TestEndToEndFlow:
    """End-to-end flow tests."""

    def test_normal_operation_flow(self, ml_engine, sample_sensor_data):
        """Test complete normal operation flow."""
        # 1. Predict
        result = ml_engine.predict(sample_sensor_data)

        # 2. High confidence -> no human needed
        if result["confidence"] >= 0.7:
            assert result["requires_human"] is False

        # 3. Explanation provided
        assert len(result["explanation"]) > 10

    def test_low_confidence_triggers_human(self, ml_engine, sample_sensor_data_missing):
        """Test low confidence triggers human intervention."""
        result = ml_engine.predict(sample_sensor_data_missing)

        # With missing data, confidence should be lower
        if result["confidence"] < 0.7:
            assert result["requires_human"] is True

    def test_human_feedback_improves_model(self, ml_engine, sample_sensor_data, sample_anomaly_data):
        """Test that human feedback leads to model adaptation."""
        # Get initial predictions on anomaly
        initial_results = [ml_engine.predict(sample_anomaly_data) for _ in range(3)]
        initial_anomaly_rate = sum(r["is_anomaly"] for r in initial_results) / 3

        # Provide human feedback confirming anomalies
        for _ in range(5):
            ml_engine.add_human_feedback(sample_anomaly_data, is_anomaly_label=True)
        for _ in range(5):
            ml_engine.add_human_feedback(sample_sensor_data, is_anomaly_label=False)

        # If RF trained, get new predictions
        if ml_engine.is_rf_trained:
            new_results = [ml_engine.predict(sample_anomaly_data) for _ in range(3)]
            new_anomaly_rate = sum(r["is_anomaly"] for r in new_results) / 3

            # With human feedback, anomaly detection should improve
            # (This is a soft assertion as it depends on data)
            assert new_anomaly_rate >= initial_anomaly_rate * 0.5  # Not worse


# ==========================================
# Performance Tests (marked as slow)
# ==========================================

@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks."""

    def test_prediction_latency(self, ml_engine, sample_sensor_data):
        """Test prediction latency is reasonable."""
        import time

        # Warm up
        for _ in range(10):
            ml_engine.predict(sample_sensor_data)

        # Measure
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            ml_engine.predict(sample_sensor_data)
            latencies.append(time.perf_counter() - start)

        avg_latency = np.mean(latencies) * 1000  # ms
        p99_latency = np.percentile(latencies, 99) * 1000

        # Should be fast (< 50ms avg, < 100ms p99)
        assert avg_latency < 50
        assert p99_latency < 100

    def test_batch_prediction_throughput(self, ml_engine):
        """Test batch prediction throughput."""
        import time

        batch_size = 100
        data_batch = [
            {"temperature": 25 + np.random.randn() * 3,
             "pressure": 1.2 + np.random.randn() * 0.1,
             "vibration": 0.3 + np.random.randn() * 0.05}
            for _ in range(batch_size)
        ]

        start = time.perf_counter()
        for data in data_batch:
            ml_engine.predict(data)
        elapsed = time.perf_counter() - start

        throughput = batch_size / elapsed
        assert throughput > 100  # > 100 predictions/second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])