"""
Monitoring and Observability Module for STAMPER_TSLR.

Provides:
- Structured JSON logging with correlation IDs
- Prometheus metrics exposition
- OpenTelemetry distributed tracing
- Health and readiness checks
"""

import contextvars
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_monitoring_config

# ==========================================
# Correlation ID Context
# ==========================================

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(cid)


# ==========================================
# Structured Logging Setup
# ==========================================


def configure_logging() -> None:
    """Configure structlog for JSON structured logging."""
    config = get_monitoring_config()

    # Standard library logging setup
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level.upper()),
    )

    # Structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
    ]

    if config.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# ==========================================
# Prometheus Metrics
# ==========================================

# Create custom registry to avoid conflicts
METRICS_REGISTRY = CollectorRegistry()

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=METRICS_REGISTRY,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=METRICS_REGISTRY,
)

# ML Prediction metrics
predictions_total = Counter(
    "predictions_total",
    "Total predictions made",
    ["prediction", "model_version"],
    registry=METRICS_REGISTRY,
)

prediction_confidence = Histogram(
    "prediction_confidence",
    "Prediction confidence scores",
    ["prediction"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=METRICS_REGISTRY,
)

prediction_uncertainty_epistemic = Histogram(
    "prediction_uncertainty_epistemic",
    "Epistemic (model) uncertainty",
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=METRICS_REGISTRY,
)

prediction_uncertainty_aleatoric = Histogram(
    "prediction_uncertainty_aleatoric",
    "Aleatoric (data) uncertainty",
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=METRICS_REGISTRY,
)

# Anomaly metrics
anomalies_detected_total = Counter(
    "anomalies_detected_total",
    "Total anomalies detected",
    ["detector"],
    registry=METRICS_REGISTRY,
)

# Human intervention metrics
interventions_total = Counter(
    "interventions_total",
    "Total human interventions",
    ["action"],  # approved, overridden
    registry=METRICS_REGISTRY,
)

intervention_queue_depth = Gauge(
    "intervention_queue_depth",
    "Current number of items in intervention queue",
    registry=METRICS_REGISTRY,
)

# Drift detection metrics
drift_detected_total = Counter(
    "drift_detected_total",
    "Total drift events detected",
    ["drift_type"],  # data, concept, prediction
    registry=METRICS_REGISTRY,
)

drift_severity = Gauge(
    "drift_severity",
    "Current drift severity score",
    ["drift_type"],
    registry=METRICS_REGISTRY,
)

# Model retraining metrics
model_retrains_total = Counter(
    "model_retrains_total",
    "Total model retraining events",
    ["trigger"],  # drift, feedback, scheduled
    registry=METRICS_REGISTRY,
)

model_training_duration_seconds = Histogram(
    "model_training_duration_seconds",
    "Model training duration in seconds",
    ["model_type"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0),
    registry=METRICS_REGISTRY,
)

# Data quality metrics
missing_data_ratio = Gauge(
    "missing_data_ratio",
    "Ratio of missing features in incoming data",
    ["feature"],
    registry=METRICS_REGISTRY,
)

# WebSocket metrics
ws_connections_active = Gauge(
    "ws_connections_active",
    "Active WebSocket connections",
    registry=METRICS_REGISTRY,
)

ws_messages_total = Counter(
    "ws_messages_total",
    "Total WebSocket messages",
    ["direction"],  # sent, received
    registry=METRICS_REGISTRY,
)

# System metrics
system_info = Gauge(
    "system_info",
    "System information",
    ["version", "environment"],
    registry=METRICS_REGISTRY,
)


def record_system_info(version: str, environment: str) -> None:
    """Record system metadata."""
    system_info.labels(version=version, environment=environment).set(1)


# ==========================================
# OpenTelemetry Tracing
# ==========================================

_tracer = None
_tracing_initialized = False


def init_tracing() -> None:
    """Initialize OpenTelemetry tracing."""
    global _tracer, _tracing_initialized

    if _tracing_initialized:
        return

    config = get_monitoring_config()
    if not config.tracing_enabled:
        _tracing_initialized = True
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({SERVICE_NAME: config.tracing_service_name})
        provider = TracerProvider(resource=resource)

        if config.tracing_endpoint:
            exporter = OTLPSpanExporter(endpoint=config.tracing_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
        _tracing_initialized = True
    except ImportError:
        # OpenTelemetry not installed
        _tracing_initialized = True


def get_tracer():
    """Get the OpenTelemetry tracer."""
    if not _tracing_initialized:
        init_tracing()
    return _tracer


def trace_span(name: str, attributes: dict[str, Any] = None):
    """Decorator to trace a function with OpenTelemetry."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)

            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, str(v))
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, str(v))
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    span.record_exception(e)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ==========================================
# FastAPI Middleware
# ==========================================


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring: logging, metrics, correlation IDs."""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = get_logger("http")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Generate or extract correlation ID
        correlation_id = request.headers.get(
            get_monitoring_config().correlation_id_header, str(uuid.uuid4())[:8]
        )
        set_correlation_id(correlation_id)

        # Bind correlation ID to structlog context
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        # Start timing
        start_time = time.perf_counter()

        # Log request
        self.logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            client_host=request.client.host if request.client else None,
        )

        # Process request
        response = None
        try:
            response = await call_next(request)
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration * 1000, 2),
                error=str(e),
                exc_info=True,
            )
            # Record error metric
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500,
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)
            raise
        finally:
            # Calculate duration
            duration = time.perf_counter() - start_time

            # Record metrics
            status_code = response.status_code if response else 500
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=status_code,
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)

            # Log response
            self.logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            )

            # Add correlation ID to response headers
            if response:
                response.headers["X-Correlation-ID"] = correlation_id

        return response


# ==========================================
# Health Checks
# ==========================================


class HealthChecker:
    """Health and readiness check manager."""

    def __init__(self):
        self.checks: dict[str, Callable[[], Awaitable[bool]]] = {}
        self.logger = get_logger("health")

    def register(self, name: str, check: Callable[[], Awaitable[bool]]) -> None:
        """Register a health check."""
        self.checks[name] = check

    async def check_live(self) -> dict[str, Any]:
        """Liveness probe - is the process alive?"""
        return {"status": "alive", "timestamp": time.time()}

    async def check_ready(self) -> dict[str, Any]:
        """Readiness probe - can the service handle requests?"""
        results = {}
        all_healthy = True

        for name, check in self.checks.items():
            try:
                start = time.perf_counter()
                healthy = await check()
                duration_ms = round((time.perf_counter() - start) * 1000, 2)

                results[name] = {
                    "status": "healthy" if healthy else "unhealthy",
                    "duration_ms": duration_ms,
                }
                if not healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                }
                all_healthy = False

        return {
            "status": "ready" if all_healthy else "not_ready",
            "timestamp": time.time(),
            "checks": results,
        }


# Global health checker instance
health_checker = HealthChecker()


def register_health_check(name: str, check: Callable[[], Awaitable[bool]]) -> None:
    """Register a health check function."""
    health_checker.register(name, check)


# ==========================================
# Metrics Endpoint
# ==========================================


async def metrics_endpoint() -> Response:
    """Prometheus metrics exposition endpoint."""
    config = get_monitoring_config()
    if not config.metrics_enabled:
        return Response(content="Metrics disabled", status_code=404)

    output = generate_latest(METRICS_REGISTRY)
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)


# ==========================================
# Utility Functions for Recording Metrics
# ==========================================


def record_prediction(
    prediction: str,
    confidence: float,
    epistemic_uncertainty: float = None,
    aleatoric_uncertainty: float = None,
    model_version: str = "unknown",
) -> None:
    """Record prediction metrics."""
    predictions_total.labels(prediction=prediction, model_version=model_version).inc()
    prediction_confidence.labels(prediction=prediction).observe(confidence)

    if epistemic_uncertainty is not None:
        prediction_uncertainty_epistemic.observe(epistemic_uncertainty)
    if aleatoric_uncertainty is not None:
        prediction_uncertainty_aleatoric.observe(aleatoric_uncertainty)


def record_anomaly(detector: str) -> None:
    """Record anomaly detection."""
    anomalies_detected_total.labels(detector=detector).inc()


def record_intervention(action: str) -> None:
    """Record human intervention."""
    interventions_total.labels(action=action).inc()


def set_intervention_queue_depth(depth: int) -> None:
    """Set intervention queue depth gauge."""
    intervention_queue_depth.set(depth)


def record_drift(drift_type: str, severity: float) -> None:
    """Record drift detection event."""
    drift_detected_total.labels(drift_type=drift_type).inc()
    drift_severity.labels(drift_type=drift_type).set(severity)


def record_retrain(trigger: str, duration_seconds: float, model_type: str) -> None:
    """Record model retraining."""
    model_retrains_total.labels(trigger=trigger).inc()
    model_training_duration_seconds.labels(model_type=model_type).observe(
        duration_seconds
    )


def record_missing_data(feature: str, ratio: float) -> None:
    """Record missing data ratio for a feature."""
    missing_data_ratio.labels(feature=feature).set(ratio)


def record_ws_connection(delta: int) -> None:
    """Update active WebSocket connection count."""
    ws_connections_active.inc(delta)


def record_ws_message(direction: str) -> None:
    """Record WebSocket message."""
    ws_messages_total.labels(direction=direction).inc()
