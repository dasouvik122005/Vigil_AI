"""
Configuration management for STAMPER_TSLR Adaptive Decision Intelligence Platform.

Uses Pydantic Settings for type-safe configuration from environment variables
and YAML files. Supports dev/staging/prod profiles.
"""

import os
from functools import lru_cache
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """ML Model hyperparameters and settings."""

    # IsolationForest
    iso_contamination: float = Field(
        default=0.05, description="Expected proportion of outliers"
    )
    iso_n_estimators: int = Field(default=100, description="Number of base estimators")
    iso_max_samples: str = Field(
        default="auto", description="Number of samples to draw"
    )
    iso_random_state: int = Field(default=42, description="Random seed")

    # RandomForest (supervised adaptation)
    rf_n_estimators: int = Field(default=100, description="Number of trees")
    rf_max_depth: int | None = Field(default=10, description="Max depth of trees")
    rf_min_samples_split: int = Field(default=5, description="Min samples to split")
    rf_min_samples_leaf: int = Field(default=2, description="Min samples per leaf")
    rf_random_state: int = Field(default=42, description="Random seed")

    # Ensemble
    ensemble_enabled: bool = Field(
        default=True, description="Enable algorithm ensemble"
    )
    ensemble_voting: str = Field(
        default="soft", description="Voting strategy: soft/hard/weighted"
    )
    ensemble_weights: list[float] | None = Field(
        default=None, description="Custom weights per algorithm"
    )

    # Uncertainty Quantification
    uncertainty_method: str = Field(
        default="ensemble", description="Method: ensemble/conformal/bayesian"
    )
    epistemic_samples: int = Field(
        default=10, description="MC dropout samples for epistemic uncertainty"
    )
    conformal_calibration_size: int = Field(
        default=500, description="Calibration set size for conformal prediction"
    )
    confidence_threshold: float = Field(
        default=0.70, description="Threshold for human intervention"
    )

    # Drift Detection
    drift_enabled: bool = Field(
        default=True, description="Enable concept drift detection"
    )
    drift_window_size: int = Field(
        default=100, description="Sliding window size for drift detection"
    )
    drift_sensitivity: float = Field(
        default=0.05, description="Statistical significance threshold"
    )
    drift_min_samples: int = Field(
        default=50, description="Minimum samples before drift check"
    )
    retrain_on_drift: bool = Field(
        default=True, description="Auto-retrain when drift detected"
    )
    retrain_min_samples: int = Field(
        default=200, description="Minimum samples for retraining"
    )

    # Human Feedback
    feedback_batch_size: int = Field(
        default=5, description="Human feedback samples before retrain"
    )
    feedback_max_buffer: int = Field(
        default=1000, description="Max feedback buffer size"
    )


class DataConfig(BaseSettings):
    """Data processing and generation settings."""

    # Sensor data
    sensor_features: list[str] = Field(
        default=["temperature", "pressure", "vibration"],
        description="Expected sensor feature names",
    )
    sensor_missing_prob: float = Field(
        default=0.25, description="Probability of missing sensor data (Hard Mode)"
    )

    # Multimodal
    multimodal_enabled: bool = Field(
        default=False, description="Enable multimodal processing"
    )
    max_sequence_length: int = Field(
        default=100, description="Max time-series sequence length"
    )
    text_max_length: int = Field(default=512, description="Max text token length")
    image_size: list[int] = Field(
        default=[224, 224], description="Image resize dimensions"
    )

    # Encoders
    text_encoder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Text embedding model",
    )
    image_encoder_model: str = Field(
        default="resnet50", description="Image encoder backbone"
    )
    timeseries_encoder: str = Field(
        default="patchtst", description="Time-series encoder: patchtst/lstm/cnn"
    )

    # Fusion
    fusion_strategy: str = Field(
        default="late", description="Fusion: early/late/cross_attention"
    )
    fusion_hidden_dim: int = Field(
        default=256, description="Fusion layer hidden dimension"
    )


class APIConfig(BaseSettings):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, description="Port number")
    workers: int = Field(default=4, description="Number of worker processes")
    reload: bool = Field(default=True, description="Enable auto-reload (dev)")

    # CORS
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_credentials: bool = Field(default=True, description="Allow credentials")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window: int = Field(default=60, description="Window in seconds")

    # WebSocket
    ws_enabled: bool = Field(default=True, description="Enable WebSocket endpoints")
    ws_heartbeat_interval: int = Field(
        default=30, description="Heartbeat interval in seconds"
    )
    ws_max_connections: int = Field(
        default=1000, description="Max concurrent WebSocket connections"
    )

    # API Versioning
    api_version: str = Field(default="v1", description="API version prefix")
    api_prefix: str = Field(default="/api", description="API route prefix")


class MonitoringConfig(BaseSettings):
    """Observability and monitoring configuration."""

    # Logging
    log_level: str = Field(
        default="INFO", description="Log level: DEBUG/INFO/WARNING/ERROR"
    )
    log_format: str = Field(default="json", description="Log format: json/text")
    log_file: str | None = Field(
        default=None, description="Log file path (None = stdout)"
    )
    correlation_id_header: str = Field(
        default="X-Correlation-ID", description="Header for request correlation"
    )

    # Metrics (Prometheus)
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics exposition port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")

    # Tracing (OpenTelemetry)
    tracing_enabled: bool = Field(
        default=False, description="Enable distributed tracing"
    )
    tracing_endpoint: str | None = Field(
        default=None, description="OTLP endpoint (e.g., jaeger:4317)"
    )
    tracing_service_name: str = Field(
        default="stamper-tslr", description="Service name for tracing"
    )
    tracing_sample_rate: float = Field(default=0.1, description="Trace sampling rate")

    # Health checks
    health_check_interval: int = Field(
        default=30, description="Health check interval in seconds"
    )
    dependency_timeout: float = Field(
        default=5.0, description="Dependency check timeout"
    )


class StorageConfig(BaseSettings):
    """Data persistence configuration."""

    # Database
    database_url: str = Field(
        default="sqlite:///./stamper_tslr.db", description="SQLAlchemy database URL"
    )
    database_pool_size: int = Field(default=10, description="Connection pool size")
    database_max_overflow: int = Field(
        default=20, description="Max overflow connections"
    )
    database_echo: bool = Field(default=False, description="Echo SQL queries")

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    redis_max_connections: int = Field(default=50, description="Max Redis connections")
    redis_socket_timeout: float = Field(default=5.0, description="Socket timeout")

    # Object storage (for model artifacts)
    storage_backend: str = Field(default="local", description="Storage: local/s3/minio")
    storage_local_path: str = Field(
        default="./models", description="Local model storage path"
    )
    storage_s3_bucket: str | None = Field(default=None, description="S3 bucket name")
    storage_s3_region: str | None = Field(default=None, description="S3 region")
    storage_s3_endpoint: str | None = Field(
        default=None, description="S3 endpoint (for MinIO)"
    )


class FrontendConfig(BaseSettings):
    """Frontend configuration."""

    dev_server_port: int = Field(default=5173, description="Vite dev server port")
    dev_server_host: str = Field(
        default="localhost", description="Vite dev server host"
    )
    api_base_url: str = Field(
        default="http://localhost:8000", description="Backend API base URL"
    )
    ws_base_url: str = Field(
        default="ws://localhost:8000", description="WebSocket base URL"
    )


class Settings(BaseSettings):
    """Main settings class combining all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: str = Field(
        default="development", description="Environment: development/staging/production"
    )
    debug: bool = Field(default=True, description="Debug mode")
    secret_key: str = Field(
        default="dev-secret-change-in-production",
        description="Secret key for sessions/JWT",
    )

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    def get_api_prefix(self) -> str:
        return f"{self.api.api_prefix}/{self.api.api_version}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Supports loading from YAML config file in addition to environment variables.
    Priority: env vars > YAML file > defaults
    """
    # Check for YAML config file
    config_path = os.environ.get("STAMPER_CONFIG_PATH") or "config.yaml"

    yaml_config: dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    # Create settings with YAML as base, env vars will override
    settings = Settings(**yaml_config)
    return settings


# Convenience function for getting typed config sections
def get_model_config() -> ModelConfig:
    return get_settings().model


def get_data_config() -> DataConfig:
    return get_settings().data


def get_api_config() -> APIConfig:
    return get_settings().api


def get_monitoring_config() -> MonitoringConfig:
    return get_settings().monitoring


def get_storage_config() -> StorageConfig:
    return get_settings().storage


def get_frontend_config() -> FrontendConfig:
    return get_settings().frontend
