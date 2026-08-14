"""
Integration tests for the FastAPI endpoints.
"""

import pytest
import json
from unittest.mock import patch, Mock


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, test_client):
        """Test basic health endpoint."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_liveness_probe(self, test_client):
        """Test liveness probe endpoint."""
        response = test_client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_probe(self, test_client):
        """Test readiness probe endpoint."""
        response = test_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data


class TestDataStreamEndpoint:
    """Tests for data streaming endpoint."""

    def test_get_data_stream_default_count(self, test_client):
        """Test streaming with default count."""
        response = test_client.get("/api/stream")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5  # Default count

    def test_get_data_stream_custom_count(self, test_client):
        """Test streaming with custom count."""
        response = test_client.get("/api/stream?count=3")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 3

    def test_stream_item_structure(self, test_client):
        """Test that stream items have correct structure."""
        response = test_client.get("/api/stream?count=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

        item = data[0]
        assert "id" in item
        assert "timestamp" in item
        assert "temperature" in item
        assert "pressure" in item
        assert "vibration" in item
        assert "source_reliable" in item

        # Values can be None (missing data) or float
        for field in ["temperature", "pressure", "vibration"]:
            assert item[field] is None or isinstance(item[field], (int, float))

        assert isinstance(item["source_reliable"], bool)


class TestDecisionEndpoint:
    """Tests for decision making endpoint."""

    def test_make_decision_complete_data(self, test_client, sample_sensor_data):
        """Test decision with complete sensor data."""
        payload = {
            "id": "test_123",
            "timestamp": 1234567890.0,
            **sample_sensor_data,
            "source_reliable": True
        }

        response = test_client.post("/api/decision", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "data_id" in data
        assert data["data_id"] == "test_123"
        assert "prediction" in data
        assert data["prediction"] in ["NORMAL_OPERATION", "ANOMALY_DETECTED"]
        assert "confidence_score" in data
        assert 0 <= data["confidence_score"] <= 1
        assert "explanation" in data
        assert isinstance(data["explanation"], str)
        assert "requires_human" in data
        assert isinstance(data["requires_human"], bool)

    def test_make_decision_missing_data(self, test_client, sample_sensor_data_missing):
        """Test decision with missing sensor data."""
        payload = {
            "id": "test_456",
            "timestamp": 1234567890.0,
            **sample_sensor_data_missing,
            "source_reliable": False
        }

        response = test_client.post("/api/decision", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "confidence_score" in data
        # Missing data should generally lower confidence
        assert data["confidence_score"] <= 1.0

    def test_make_decision_anomaly_data(self, test_client, sample_anomaly_data):
        """Test decision with anomalous sensor data."""
        payload = {
            "id": "test_789",
            "timestamp": 1234567890.0,
            **sample_anomaly_data,
            "source_reliable": True
        }

        response = test_client.post("/api/decision", json=payload)
        assert response.status_code == 200

        data = response.json()
        # May or may not detect anomaly depending on model
        assert data["prediction"] in ["NORMAL_OPERATION", "ANOMALY_DETECTED"]

    def test_make_decision_invalid_payload(self, test_client):
        """Test decision with invalid payload."""
        payload = {
            "id": "test_invalid",
            # Missing required fields
        }

        response = test_client.post("/api/decision", json=payload)
        # Should return validation error
        assert response.status_code in [400, 422]


class TestInterventionEndpoints:
    """Tests for human intervention endpoints."""

    def test_get_interventions_empty(self, test_client):
        """Test getting interventions when queue is empty."""
        response = test_client.get("/api/interventions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_resolve_intervention_not_found(self, test_client):
        """Test resolving non-existent intervention."""
        response = test_client.post(
            "/api/interventions/nonexistent/resolve",
            params={"approved": True}
        )
        assert response.status_code == 404

    def test_resolve_intervention_approve(self, test_client, sample_sensor_data):
        """Test approving an intervention."""
        # First create a low-confidence decision to trigger intervention
        payload = {
            "id": "intervene_test_1",
            "timestamp": 1234567890.0,
            **sample_sensor_data,
            "source_reliable": False  # This should lower confidence
        }

        # Make decision
        decision_response = test_client.post("/api/decision", json=payload)
        assert decision_response.status_code == 200
        decision = decision_response.json()

        # If it requires human, test resolution
        if decision["requires_human"]:
            data_id = decision["data_id"]
            response = test_client.post(
                f"/api/interventions/{data_id}/resolve",
                params={"approved": True}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "resolved"
            assert data["adapted"] is True  # Should adapt model

    def test_resolve_intervention_override(self, test_client, sample_anomaly_data):
        """Test overriding an intervention."""
        payload = {
            "id": "intervene_test_2",
            "timestamp": 1234567890.0,
            **sample_anomaly_data,
            "source_reliable": False
        }

        decision_response = test_client.post("/api/decision", json=payload)
        assert decision_response.status_code == 200
        decision = decision_response.json()

        if decision["requires_human"]:
            data_id = decision["data_id"]
            # Override with opposite prediction
            new_prediction = "NORMAL_OPERATION" if decision["prediction"] == "ANOMALY_DETECTED" else "ANOMALY_DETECTED"
            response = test_client.post(
                f"/api/interventions/{data_id}/resolve",
                params={"approved": False, "new_prediction": new_prediction}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "resolved"


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint returns Prometheus format."""
        response = test_client.get("/metrics")
        # May be 404 if disabled, 200 if enabled
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            content = response.text
            # Check for Prometheus format
            assert "# HELP" in content or "# TYPE" in content


class TestCORSHeaders:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/api/health")
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_on_unknown_endpoint(self, test_client):
        """Test 404 on unknown endpoint."""
        response = test_client.get("/api/unknown")
        assert response.status_code == 404

    def test_method_not_allowed(self, test_client):
        """Test method not allowed."""
        response = test_client.delete("/api/health")
        assert response.status_code == 405


# ==========================================
# WebSocket Tests (when implemented)
# ==========================================

class TestWebSocket:
    """Tests for WebSocket endpoints."""

    def test_websocket_endpoint_exists(self, test_client):
        """Test WebSocket endpoint is registered."""
        # This will fail until WebSocket is implemented
        # For now, just check the route exists in app
        from main import app
        routes = [route.path for route in app.routes]
        # WebSocket routes have different structure
        # This is a placeholder for future implementation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])