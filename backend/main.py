"""
STAMPER_TSLR: Adaptive Decision Intelligence Platform - FastAPI Backend.

This module provides the main API endpoints for:
- Data streaming with missing data simulation (Hard Mode)
- Real-time anomaly detection with confidence scoring
- Human-in-the-loop intervention queue
- Model adaptation via human feedback
- Health checks and Prometheus metrics
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Import our modules
from config import get_api_config, get_monitoring_config, get_settings
from data_generator import DataStreamer
from ml_engine import AdaptiveDecisionEngine
from monitoring import (
    MonitoringMiddleware,
    configure_logging,
    get_logger,
    health_checker,
    metrics_endpoint,
    record_intervention,
    record_missing_data,
    record_prediction,
    record_system_info,
    register_health_check,
    set_intervention_queue_depth,
)

# ==========================================
# Configuration & Initialization
# ==========================================

settings = get_settings()
api_config = get_api_config()
monitoring_config = get_monitoring_config()

# Configure structured logging
configure_logging()
logger = get_logger("main")

# Record system info for metrics
record_system_info(version="1.0.0", environment=settings.environment)


# ==========================================
# Pydantic Models
# ==========================================


class MultimodalData(BaseModel):
    """Single sensor data point with optional missing values."""

    id: str
    timestamp: float
    temperature: float | None = None
    pressure: float | None = None
    vibration: float | None = None
    source_reliable: bool = True


class DecisionResponse(BaseModel):
    """AI Decision response with confidence and explanation."""

    data_id: str
    prediction: str = Field(description="NORMAL_OPERATION or ANOMALY_DETECTED")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence in prediction"
    )
    explanation: str = Field(description="Human-readable explanation")
    requires_human: bool = Field(description="Whether human intervention is required")


class InterventionResolveRequest(BaseModel):
    """Request to resolve a human intervention."""

    approved: bool
    new_prediction: str | None = None


# ==========================================
# Application Lifecycle
# ==========================================

# Initialize components
streamer = DataStreamer(filename="sensor_data.csv")
ai_engine = AdaptiveDecisionEngine(historical_data_path="sensor_data.csv")

# In-memory storage (will be replaced with persistent storage in Phase 6)
intervention_queue: list[dict[str, Any]] = []
historical_decisions: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("application_starting", environment=settings.environment)

    # Register health checks
    register_health_check("ml_engine", lambda: _check_ml_engine())
    register_health_check("data_streamer", lambda: _check_data_streamer())

    yield

    # Shutdown
    logger.info("application_shutting_down")


def _check_ml_engine() -> bool:
    """Health check for ML engine."""
    try:
        # Quick prediction to verify engine works
        test_data = {"temperature": 25.0, "pressure": 1.2, "vibration": 0.3}
        result = ai_engine.predict(test_data)
        return "is_anomaly" in result and "confidence" in result
    except Exception:
        return False


def _check_data_streamer() -> bool:
    """Health check for data streamer."""
    try:
        batch = streamer.get_next_batch(count=1)
        return len(batch) > 0
    except Exception:
        return False


# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="Adaptive Decision Intelligence API",
    description="AI-powered anomaly detection with human-in-the-loop adaptation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_credentials=api_config.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware)


# ==========================================
# Health & Metrics Endpoints
# ==========================================


@app.get("/api/health", tags=["Health"])
def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/health/live", tags=["Health"])
async def liveness_probe():
    """Kubernetes liveness probe."""
    return await health_checker.check_live()


@app.get("/health/ready", tags=["Health"])
async def readiness_probe():
    """Kubernetes readiness probe."""
    return await health_checker.check_ready()


@app.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics exposition endpoint."""
    return await metrics_endpoint()


# ==========================================
# Data Streaming Endpoint
# ==========================================


@app.get("/api/stream", response_model=list[MultimodalData], tags=["Data"])
def get_data_stream(count: int = 5):
    """
    Stream realistic sensor data from CSV.
    Implements Hard Mode: missing fields are handled dynamically.
    """
    raw_batch = streamer.get_next_batch(count=count)

    stream = []
    for raw in raw_batch:
        # Track missing data ratio for monitoring
        missing_count = sum(
            1
            for v in [raw["temperature"], raw["pressure"], raw["vibration"]]
            if v is None
        )
        for feat, val in [
            ("temperature", raw["temperature"]),
            ("pressure", raw["pressure"]),
            ("vibration", raw["vibration"]),
        ]:
            if val is None:
                record_missing_data(feat, 1.0)
            else:
                record_missing_data(feat, 0.0)

        data = MultimodalData(
            id=f"evt_{int(time.time()*1000)}_{len(stream)}_{uuid.uuid4().hex[:4]}",
            timestamp=time.time(),
            temperature=raw["temperature"],
            pressure=raw["pressure"],
            vibration=raw["vibration"],
            source_reliable=not raw["is_corrupted"],
        )
        stream.append(data)

    return stream


# ==========================================
# Decision Making Endpoint
# ==========================================


@app.post("/api/decision", response_model=DecisionResponse, tags=["Decision"])
def make_decision(data: MultimodalData):
    """
    Make a decision using the ML Engine.
    Returns prediction, confidence, explanation, and human intervention flag.
    """
    data_dict = {
        "temperature": data.temperature,
        "pressure": data.pressure,
        "vibration": data.vibration,
    }

    # Store raw data for future human feedback
    historical_decisions[data.id] = data_dict

    # Run ML inference
    result = ai_engine.predict(data_dict)

    prediction = "ANOMALY_DETECTED" if result["is_anomaly"] else "NORMAL_OPERATION"

    # Determine if human intervention needed based on confidence threshold
    confidence_threshold = settings.model.confidence_threshold
    requires_human = result["confidence"] < confidence_threshold

    response = DecisionResponse(
        data_id=data.id,
        prediction=prediction,
        confidence_score=round(result["confidence"], 2),
        explanation=result["explanation"],
        requires_human=requires_human,
    )

    # Record metrics
    record_prediction(
        prediction=prediction,
        confidence=result["confidence"],
        model_version="1.0.0",  # TODO: Get from model registry
    )

    if requires_human:
        intervention_queue.append(response.model_dump())
        set_intervention_queue_depth(len(intervention_queue))
        logger.info(
            "intervention_required",
            data_id=data.id,
            confidence=result["confidence"],
            threshold=confidence_threshold,
            queue_depth=len(intervention_queue),
        )

    return response


# ==========================================
# Human Intervention Endpoints
# ==========================================


@app.get("/api/interventions", tags=["Intervention"])
def get_intervention_queue():
    """Get all pending human interventions."""
    return intervention_queue


@app.post("/api/interventions/{data_id}/resolve", tags=["Intervention"])
def resolve_intervention(data_id: str, request: InterventionResolveRequest):
    """
    Resolve a human intervention and feed feedback back to the AI engine.
    This enables the model to adapt from human corrections.
    """
    global intervention_queue

    # Find intervention in queue
    intervention = None
    for item in intervention_queue:
        if item["data_id"] == data_id:
            intervention = item
            break

    if intervention is None:
        raise HTTPException(
            status_code=404, detail="Item not found in intervention queue"
        )

    # Remove from queue
    intervention_queue = [x for x in intervention_queue if x["data_id"] != data_id]
    set_intervention_queue_depth(len(intervention_queue))

    # Human Feedback Loop (Adaptation)
    raw_data = historical_decisions.get(data_id)
    adapted = False

    if raw_data:
        # Determine final decision based on human input
        if request.approved:
            # Human agrees with AI prediction
            final_decision = intervention["prediction"]
        else:
            # Human overrides with new prediction
            final_decision = request.new_prediction
            if final_decision is None:
                raise HTTPException(
                    status_code=400, detail="new_prediction required when overriding"
                )

        is_anomaly_label = final_decision == "ANOMALY_DETECTED"

        # Send back to ML Engine to learn
        ai_engine.add_human_feedback(raw_data, is_anomaly_label)
        adapted = True

        # Record intervention metrics
        action = "approved" if request.approved else "overridden"
        record_intervention(action)

        logger.info(
            "intervention_resolved",
            data_id=data_id,
            action=action,
            adapted=adapted,
            final_decision=final_decision,
        )

    return {"status": "resolved", "data_id": data_id, "adapted": adapted}


# ==========================================
# Main Entry Point
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.reload,
        workers=1 if api_config.reload else api_config.workers,
    )
