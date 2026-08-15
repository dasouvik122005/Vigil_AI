"""
Pytest configuration and shared fixtures for Vigil AI tests.
"""

import sys
import os
import pytest
import asyncio
import numpy as np
import pandas as pd
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from httpx import AsyncClient


# ==========================================
# Event Loop
# ==========================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==========================================
# Test Data Fixtures
# ==========================================

@pytest.fixture
def sample_sensor_data() -> dict:
    """Sample sensor data for testing."""
    return {
        "temperature": 25.0,
        "pressure": 1.2,
        "vibration": 0.3,
    }


@pytest.fixture
def sample_sensor_data_missing() -> dict:
    """Sample sensor data with missing values."""
    return {
        "temperature": 25.0,
        "pressure": None,
        "vibration": 0.3,
    }


@pytest.fixture
def sample_anomaly_data() -> dict:
    """Sample anomalous sensor data."""
    return {
        "temperature": 55.0,  # High temperature anomaly
        "pressure": 2.8,      # High pressure anomaly
        "vibration": 2.5,     # High vibration anomaly
    }


@pytest.fixture
def training_data() -> pd.DataFrame:
    """Generate synthetic training data."""
    np.random.seed(42)
    n_samples = 1000

    # Normal data
    temp = np.random.normal(loc=25, scale=3, size=n_samples)
    pressure = np.random.normal(loc=1.2, scale=0.1, size=n_samples)
    vibration = np.random.normal(loc=0.3, scale=0.05, size=n_samples)

    df = pd.DataFrame({
        'temperature': temp,
        'pressure': pressure,
        'vibration': vibration
    })

    # Inject anomalies (5%)
    n_anomalies = int(n_samples * 0.05)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['temp_high', 'pressure_high', 'vibration_high', 'all_high'])
        if anomaly_type in ['temp_high', 'all_high']:
            df.loc[idx, 'temperature'] = np.random.uniform(45, 60)
        if anomaly_type in ['pressure_high', 'all_high']:
            df.loc[idx, 'pressure'] = np.random.uniform(2.0, 3.0)
        if anomaly_type in ['vibration_high', 'all_high']:
            df.loc[idx, 'vibration'] = np.random.uniform(1.2, 2.5)

    return df


@pytest.fixture
def mock_feedback_buffer() -> list:
    """Mock human feedback buffer."""
    return [
        (np.array([25.0, 1.2, 0.3]), 0),  # Normal
        (np.array([26.0, 1.3, 0.35]), 0),  # Normal
        (np.array([55.0, 2.8, 2.5]), 1),   # Anomaly
        (np.array([24.0, 1.1, 0.25]), 0),  # Normal
        (np.array([50.0, 2.5, 2.0]), 1),   # Anomaly
    ]


# ==========================================
# FastAPI Test Client Fixtures
# ==========================================

@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create synchronous test client."""
    # Import here to avoid circular imports
    from main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create asynchronous test client."""
    from main import app
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ==========================================
# ML Engine Fixtures
# ==========================================

@pytest.fixture
def ml_engine(training_data):
    """Create ML engine with trained models."""
    from ml_engine import AdaptiveDecisionEngine
    import tempfile

    # Save training data to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        training_data.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        engine = AdaptiveDecisionEngine(historical_data_path=temp_path)
        yield engine
    finally:
        os.unlink(temp_path)


@pytest.fixture
def ml_engine_untrained():
    """Create ML engine without training data (uses defaults)."""
    from ml_engine import AdaptiveDecisionEngine
    engine = AdaptiveDecisionEngine(historical_data_path="nonexistent.csv")
    yield engine


# ==========================================
# Configuration Fixtures
# ==========================================

@pytest.fixture
def test_settings(monkeypatch):
    """Override settings for testing."""
    from config import Settings, ModelConfig, DataConfig, APIConfig, MonitoringConfig, StorageConfig

    # Set test environment variables
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("MODEL__CONFIDENCE_THRESHOLD", "0.5")  # Lower for testing
    monkeypatch.setenv("MODEL__FEEDBACK_BATCH_SIZE", "2")     # Faster retraining
    monkeypatch.setenv("MONITORING__LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MONITORING__METRICS_ENABLED", "false")
    monkeypatch.setenv("STORAGE__DATABASE_URL", "sqlite:///./test.db")

    # Clear settings cache
    from config import get_settings
    get_settings.cache_clear()

    yield get_settings()

    # Cleanup
    get_settings.cache_clear()


# ==========================================
# Mock Fixtures
# ==========================================

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = Mock()
    mock.get = Mock(return_value=None)
    mock.set = Mock(return_value=True)
    mock.incr = Mock(return_value=1)
    mock.decr = Mock(return_value=0)
    mock.publish = Mock(return_value=1)
    mock.subscribe = Mock()
    return mock


@pytest.fixture
def mock_celery():
    """Mock Celery app."""
    mock = Mock()
    mock.send_task = Mock(return_value=Mock(id="test-task-id"))
    mock.control = Mock()
    mock.control.inspect = Mock(return_value=Mock(active=Mock(return_value={})))
    return mock


# ==========================================
# Test Helpers
# ==========================================

def assert_valid_prediction_response(response: dict) -> None:
    """Assert that a prediction response has all required fields."""
    assert "data_id" in response
    assert "prediction" in response
    assert response["prediction"] in ["NORMAL_OPERATION", "ANOMALY_DETECTED"]
    assert "confidence_score" in response
    assert 0.0 <= response["confidence_score"] <= 1.0
    assert "explanation" in response
    assert isinstance(response["explanation"], str)
    assert len(response["explanation"]) > 0
    assert "requires_human" in response
    assert isinstance(response["requires_human"], bool)


def assert_valid_uncertainty_response(response: dict) -> None:
    """Assert that a prediction response includes uncertainty decomposition."""
    assert_valid_prediction_response(response)
    # Additional uncertainty fields (when implemented)
    # assert "uncertainty" in response
    # assert "epistemic" in response["uncertainty"]
    # assert "aleatoric" in response["uncertainty"]
    # assert "total" in response["uncertainty"]


# ==========================================
# Async Test Helpers
# ==========================================

async def wait_for_condition(condition: callable, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """Wait for a condition to become true."""
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        if await condition() if asyncio.iscoroutinefunction(condition) else condition():
            return True
        await asyncio.sleep(interval)
    return False