# STAMPER_TSLR: AI-Powered Adaptive Decision Intelligence Platform

[![CI](https://github.com/GITtridib22/STAMPER_TSLR/actions/workflows/ci.yml/badge.svg)](https://github.com/GITtridib22/STAMPER_TSLR/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node 20+](https://img.shields.io/badge/node-20+-green.svg)](https://nodejs.org/)

> **An adaptive decision intelligence platform that knows not only what to decide, but also when it should NOT trust itself.**

## Overview

STAMPER_TSLR (Spatio-Temporal Adaptive Multimodal Prediction Engine for Real-time Strategic Logic & Reasoning) is a production-grade AI platform designed for reliable decision-making in complex, uncertain, and rapidly changing environments. The system ingests heterogeneous data streams, quantifies its own uncertainty, explains its reasoning, and proactively requests human intervention when its confidence falls below acceptable thresholds.

### Core Philosophy

> **"Can you build AI that knows not only what to decide, but also when it should NOT trust itself?"**

This is the central challenge STAMPER_TSLR addresses. Unlike traditional ML systems that output point predictions with heuristic confidence scores, STAMPER_TSLR provides:

- **Epistemic Uncertainty** — Model uncertainty (what the model doesn't know)
- **Aleatoric Uncertainty** — Data uncertainty (inherent noise in observations)
- **Calibrated Prediction Intervals** — Finite-sample coverage guarantees via conformal prediction
- **Concept Drift Detection** — Automated detection of distribution shifts with champion/challenger retraining
- **Human-in-the-Loop** — Structured intervention workflow with feedback-driven model adaptation
- **Multimodal Fusion** — Sensor streams, text logs, images, time-series, and tabular data

---

## Architecture

```mermaid
flowchart TD
    subgraph Modalities[Data Sources]
        S[Sensors]
        L[Logs]
        I[Images]
        E[Events]
    end

    subgraph Encoders[Modality Encoders]
        SE[Sensor Encoder]
        TE[Text Encoder]
        IE[Image Encoder]
        TSE[Timeseries Encoder]
        TabE[Tabular Encoder]
    end

    S --> SE
    L --> TE
    I --> IE
    E --> TabE

    subgraph Fusion[Cross-Modal Fusion]
        F[Early / Late / Cross-Attention]
    end

    SE --> F
    TE --> F
    IE --> F
    TSE --> F
    TabE --> F

    subgraph AnomalyDetection[Ensemble Anomaly Detection]
        IF[IsolationForest]
        LOF[LOF]
        OCSVM[OneClassSVM]
        AE[Autoencoder]
    end

    F --> IF
    F --> LOF
    F --> OCSVM
    F --> AE

    subgraph Uncertainty[Uncertainty Quantification]
        Ep[Epistemic]
        Al[Aleatoric]
        CP[Conformal Prediction]
    end
    
    IF --> Ep
    LOF --> Ep
    OCSVM --> Al
    AE --> CP

    subgraph Decision[Decision & Explanation]
        Pred[Prediction + Calibrated Confidence]
        XAI[SHAP Feature Attributions]
    end

    Ep --> Pred
    Al --> Pred
    CP --> Pred
    Pred --> XAI

    subgraph Actions[Routing & Intervention]
        HighConf[High Confidence: Auto-Execute]
        LowConf[Low Confidence: Human Queue]
        Drift[Drift Alert: Retrain]
    end

    Pred --> HighConf
    Pred --> LowConf
    Pred --> Drift

    subgraph Feedback[Human Feedback Loop]
        Approve[Approve / Override]
        Retrain[Buffer --> Retrain]
        Deploy[Deploy New Model]
    end

    LowConf --> Approve
    Approve --> Retrain
    Retrain --> Deploy
```

---

## Features

### 🚀 Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **Multimodal Ingestion** | Sensor streams, text logs, thermal images, time-series, tabular data, discrete events | 📅 Phase 3 |
| **Ensemble Anomaly Detection** | IsolationForest, LOF, OneClassSVM, Autoencoder with uncertainty-weighted voting | 📅 Phase 2 |
| **Uncertainty Quantification** | Epistemic + Aleatoric decomposition, conformal prediction intervals | 📅 Phase 2 |
| **Explainable AI (XAI)** | SHAP local/global explanations, counterfactuals, feature dependence plots | 📅 Phase 4 |
| **Concept Drift Detection** | KS-test (data), ADWIN (concept), PSI (prediction) with automated retraining | 📅 Phase 2 |
| **Human-in-the-Loop** | Intervention queue, approve/override, feedback-driven adaptation | ⭐ MVP |
| **Real-time Streaming** | WebSocket-based live decision feed with automatic reconnection | 📅 Phase 5 |
| **Hard Mode Resilience** | Graceful degradation with 20-30% missing/corrupted data | ⭐ MVP |

### 🛠️ Platform Features

| Feature | Description |
|---------|-------------|
| **Configuration Management** | Pydantic Settings with YAML + environment variables, dev/staging/prod profiles |
| **Observability** | Structured JSON logging, Prometheus metrics, OpenTelemetry tracing, Grafana dashboards |
| **Health Checks** | Kubernetes-ready liveness/readiness probes with dependency verification |
| **Async Processing** | Celery + Redis task queue with priority routing |
| **Persistence** | PostgreSQL for decisions/feedback, Redis for caching, S3/MinIO for model artifacts |
| **Containerization** | Multi-stage Dockerfiles, docker-compose for local dev, K8s manifests |
| **CI/CD** | GitHub Actions: lint, type-check, test, build, security scan, release |
| **Testing** | Unit, integration, ML-specific (calibration, drift, robustness) tests with pytest |

---

## Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended)
- Or: **Python 3.11+**, **Node.js 20+**, **PostgreSQL 15+**, **Redis 7+**

### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/GITtridib22/STAMPER_TSLR.git
cd STAMPER_TSLR

# Start full stack (API, Worker, Frontend, DB, Redis, Prometheus, Grafana)
docker-compose up -d --build

# Verify services
curl http://localhost:8000/health/ready
curl http://localhost:5173/health
```

**Access Points:**
- **Frontend Dashboard**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)

### Option 2: Local Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
cp .env.example .env  # Customize as needed
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Option 3: Automated Setup (Windows)

If you are on Windows, you can use the provided setup script to automatically install dependencies and start both the frontend and backend in separate windows.

```powershell
# Run from the root of the project
.\setup.ps1
```

---

## API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### Health & Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Basic health check |
| `GET` | `/health/live` | Kubernetes liveness probe |
| `GET` | `/health/ready` | Kubernetes readiness probe |
| `GET` | `/metrics` | Prometheus metrics exposition |

#### Data Streaming
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stream?count=5` | Stream sensor data (simulates 25% missing) |

#### Decision Making
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/decision` | Make anomaly detection decision |

**Request:**
```json
{
  "id": "evt_123",
  "timestamp": 1699999999.0,
  "temperature": 25.5,
  "pressure": 1.2,
  "vibration": 0.3,
  "source_reliable": true
}
```

**Response:**
```json
{
  "data_id": "evt_123",
  "prediction": "NORMAL_OPERATION",
  "confidence_score": 0.92,
  "explanation": "Prediction made by IsolationForest (Unsupervised Base). All available sensor readings are within normal historical boundaries.",
  "requires_human": false
}
```

#### Human Intervention
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/interventions` | Get pending interventions |
| `POST` | `/api/interventions/{id}/resolve` | Resolve intervention |

**Resolve Request:**
```json
{
  "approved": true,
  "new_prediction": "ANOMALY_DETECTED"
}
```

#### WebSocket (Real-time)
| Endpoint | Description |
|----------|-------------|
| `WS /api/ws/stream` | Live decision stream |
| `WS /api/ws/interventions` | Live intervention notifications |

---

## Configuration

All configuration is managed via **environment variables** or **YAML config file** (`config.yaml`).

### Environment Variables (`.env`)

```bash
# Environment
ENVIRONMENT=development          # development, staging, production
DEBUG=true
SECRET_KEY=your-secret-key

# Model
MODEL__CONFIDENCE_THRESHOLD=0.70
MODEL__ENSEMBLE_ENABLED=true
MODEL__UNCERTAINTY_METHOD=ensemble
MODEL__DRIFT_ENABLED=true

# API
API__HOST=0.0.0.0
API__PORT=8000
API__WORKERS=4
API__WS_ENABLED=true

# Monitoring
MONITORING__LOG_LEVEL=INFO
MONITORING__LOG_FORMAT=json
MONITORING__METRICS_ENABLED=true
MONITORING__TRACING_ENABLED=false

# Storage
STORAGE__DATABASE_URL=postgresql://user:pass@localhost:5432/stamper
STORAGE__REDIS_URL=redis://localhost:6379/0
```

### YAML Config (`config.yaml`)

```yaml
environment: production
debug: false
secret_key: "production-secret"

model:
  confidence_threshold: 0.75
  ensemble_enabled: true
  uncertainty_method: conformal
  drift_enabled: true

api:
  workers: 8
  rate_limit_enabled: true
  rate_limit_requests: 1000

monitoring:
  log_level: INFO
  tracing_enabled: true
  tracing_endpoint: "http://jaeger:4317"
```

---

## Monitoring & Observability

### Prometheus Metrics

Key metrics exposed at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by method/endpoint/status |
| `http_request_duration_seconds` | Histogram | Request latency |
| `predictions_total` | Counter | Predictions by class and model version |
| `prediction_confidence` | Histogram | Confidence score distribution |
| `prediction_uncertainty_epistemic` | Histogram | Model uncertainty |
| `prediction_uncertainty_aleatoric` | Histogram | Data uncertainty |
| `anomalies_detected_total` | Counter | Anomalies by detector |
| `interventions_total` | Counter | Human interventions by action |
| `intervention_queue_depth` | Gauge | Pending interventions |
| `drift_detected_total` | Counter | Drift events by type |
| `drift_severity` | Gauge | Current drift severity |
| `model_retrains_total` | Counter | Retraining events by trigger |
| `ws_connections_active` | Gauge | Active WebSocket connections |

### Grafana Dashboards

Pre-configured dashboards available at http://localhost:3000:

- **STAMPER Overview** — System health, latency, predictions, anomalies, interventions, drift
- **ML Model Performance** — Confidence calibration, uncertainty decomposition, drift trends
- **Business Metrics** — Anomaly rates, human override rates, model adaptation velocity

### Structured Logging

JSON logs with correlation IDs for request tracing:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "http",
  "correlation_id": "a1b2c3d4",
  "event": "request_completed",
  "method": "POST",
  "path": "/api/v1/decision",
  "status_code": 200,
  "duration_ms": 42.5
}
```

---

## Development

### Project Structure

```
STAMPER_TSLR/
├── backend/                    # FastAPI Backend
│   ├── main.py                # API endpoints
│   ├── ml_engine.py           # Core ML logic (IsolationForest + RF)
│   ├── data_generator.py      # Synthetic data + streaming
│   ├── config.py              # Pydantic Settings configuration
│   ├── monitoring.py          # Logging, metrics, tracing, health
│   ├── uncertainty.py         # Uncertainty quantification (Phase 2)
│   ├── ensemble.py            # Algorithm ensemble (Phase 2)
│   ├── drift.py               # Drift detection (Phase 2)
│   ├── encoders/              # Modality encoders (Phase 3)
│   ├── fusion.py              # Cross-modal fusion (Phase 3)
│   ├── explain.py             # SHAP explainability (Phase 4)
│   ├── streaming.py           # WebSocket handlers (Phase 5)
│   ├── tasks.py               # Celery tasks (Phase 6)
│   ├── storage.py             # Persistence layer (Phase 6)
│   ├── models.py              # SQLAlchemy models (Phase 6)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                   # React + Vite Frontend
│   ├── src/
│   │   ├── App.jsx           # Main dashboard
│   │   ├── components/       # Reusable components
│   │   ├── hooks/            # Custom React hooks
│   │   └── utils/            # Utilities
│   ├── nginx.conf
│   ├── Dockerfile
│   └── package.json
│
├── monitoring/                 # Observability configs
│   ├── prometheus.yml
│   └── grafana/
│       ├── datasources/
│       └── dashboards/
│
├── tests/                      # Test Suite
│   ├── conftest.py            # Pytest fixtures
│   ├── test_ml_engine.py      # ML engine unit tests
│   ├── test_api.py            # API integration tests
│   └── test_ml_features.py    # ML feature contract tests
│
├── k8s/                        # Kubernetes manifests (Phase 7)
│   ├── base/
│   ├── overlays/
│   │   ├── dev/
│   │   ├── staging/
│   │   └── prod/
│
├── docker-compose.yml          # Local development stack
├── pytest.ini
├── requirements.txt
├── README.md
��── LICENSE
```

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=backend --cov-report=term-missing

# Run specific test categories
pytest tests/ -m "unit"        # Unit tests only
pytest tests/ -m "integration" # Integration tests
pytest tests/ -m "ml"          # ML-specific tests
pytest tests/ -m "slow"        # Performance benchmarks

# Frontend tests
cd frontend
npm run lint
npm run build
```

### Code Quality

```bash
# Python
ruff check backend/           # Linting
mypy backend/                 # Type checking
black backend/                # Formatting

# Frontend
cd frontend
npm run lint                  # oxlint
```

---

## Deployment

### Kubernetes (Phase 7)

```bash
# Deploy to development
kubectl apply -k k8s/overlays/dev

# Deploy to staging
kubectl apply -k k8s/overlays/staging

# Deploy to production (with canary)
kubectl apply -k k8s/overlays/prod
```

**K8s Resources:**
- `Deployment` — API (HPA), Worker (KEDA), Frontend
- `Service` — ClusterIP for API, LoadBalancer for Frontend
- `ConfigMap` / `Secret` — Configuration
- `Ingress` — TLS termination, routing
- `ServiceMonitor` — Prometheus scraping
- `HorizontalPodAutoscaler` — CPU/memory/custom metrics scaling

### Docker Images

```bash
# Build locally
docker build -t stamper-tslr-backend ./backend
docker build -t stamper-tslr-frontend ./frontend

# Or pull from GHCR
docker pull ghcr.io/gittridib22/stamper-tslr/backend:latest
docker pull ghcr.io/gittridib22/stamper-tslr/frontend:latest
```

---

## Roadmap

### ✅ Phase 1: Foundation (Complete)
- [x] Configuration management (Pydantic Settings)
- [x] Structured logging & Prometheus metrics
- [x] OpenTelemetry tracing setup
- [x] Health checks (liveness/readiness)
- [x] Test infrastructure (pytest, fixtures, CI)
- [x] Docker multi-stage builds
- [x] docker-compose local stack
- [x] GitHub Actions CI pipeline

### 🚧 Phase 2: Core ML Enhancements (In Progress)
- [ ] Uncertainty quantification (epistemic + aleatoric + conformal)
- [ ] Algorithm ensemble (IF, LOF, OC-SVM, Autoencoder)
- [ ] Concept drift detection (KS, ADWIN, PSI)
- [ ] Champion/challenger automated retraining

### 📅 Phase 3: Multimodal Support
- [ ] Extensible modality schema
- [ ] Sensor, Text, Image, Time-series, Tabular encoders
- [ ] Cross-modal fusion (early/late/cross-attention)
- [ ] Missing modality robustness

### 🔍 Phase 4: Explainability (XAI)
- [ ] SHAP integration (TreeSHAP + KernelSHAP)
- [ ] Global feature importance
- [ ] Counterfactual explanations
- [ ] Frontend visualization components

### ⚡ Phase 5: Real-time Streaming
- [ ] WebSocket endpoints
- [ ] Frontend WebSocket migration
- [ ] Connection pooling & scaling

### 💾 Phase 6: Async & Persistence
- [ ] Celery + Redis task queue
- [ ] PostgreSQL persistence
- [ ] Model versioning & artifact storage

### 🛡️ Phase 7: Production Hardening
- [ ] Kubernetes manifests (dev/staging/prod overlays)
- [ ] Canary deployments with automated rollback
- [ ] Comprehensive Grafana dashboards
- [ ] Load testing & chaos engineering
- [ ] Security hardening & compliance

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run quality checks (`ruff`, `mypy`, `pytest`)
5. Submit a Pull Request

### Code Standards

- **Python**: Black formatting, Ruff linting, MyPy strict mode
- **JavaScript/React**: ESLint + Prettier, functional components with hooks
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- **Tests**: Required for new features, maintain >80% coverage

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **scikit-learn** — Core ML algorithms
- **SHAP** — Explainable AI
- **Evidently AI** — Drift detection
- **Prometheus/Grafana** — Observability stack
- **FastAPI** — Modern Python web framework
- **React/Vite** — Frontend framework

---

## Citation

If you use STAMPER_TSLR in research, please cite:

```bibtex
@software{stamper_tslr_2024,
  title = {STAMPER_TSLR: Adaptive Decision Intelligence Platform},
  author = {STAMPER_TSLR Contributors},
  year = {2024},
  url = {https://github.com/GITtridib22/STAMPER_TSLR}
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/GITtridib22/STAMPER_TSLR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GITtridib22/STAMPER_TSLR/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for vulnerability reporting

---

<div align="center">
  <strong>Built with ❤️ for reliable AI decision-making</strong>
  <br>
  <em>"The system must quantify its confidence, explain why a decision was made, detect when its prediction may be unreliable, and request human intervention when necessary."</em>
</div>