# Vigil AI — Adaptive Decision Intelligence Platform

<div align="center">


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node 20+](https://img.shields.io/badge/node-20+-green.svg)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://react.dev/)


> **"Can you build AI that knows not only *what* to decide, but also *when* it should NOT trust itself?"**

**Vigil AI** is a production-grade, AI-powered anomaly detection platform that quantifies its own uncertainty, explains every decision, and triggers human-in-the-loop intervention when confidence is too low — so you never blindly trust a broken model.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Vigil AI (STAMPER TSLR — **S**patio-**T**emporal **A**daptive **M**ultimodal **P**rediction **E**ngine for **R**eal-time Strategic Logic and **R**easoning) is built for Hackathon Track 1: AI decision-making under uncertainty.

Industrial and critical systems produce sensor data that is frequently noisy, incomplete, or corrupted. Traditional ML pipelines either crash or silently give wrong answers. Vigil AI is designed to:

1. **Detect anomalies** using a multi-algorithm ensemble trained on historical sensor data.
2. **Quantify uncertainty** — decomposing it into epistemic (model) and aleatoric (data) components.
3. **Explain its reasoning** in plain English for every prediction.
4. **Escalate to a human** when confidence drops below a configurable threshold.
5. **Learn from corrections** — human feedback is fed back into the model in real time.
6. **Survive corrupted data** — gracefully handles 20–30% missing/corrupted fields.

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Data Ingestion"]
        CSV[CSV Sensor Data]
        API_IN[REST API Input]
    end

    subgraph HardMode["Hard Mode Resilience"]
        MIS[Missing Data Handler]
        IMP[Statistical Imputation]
    end

    subgraph MLCore["Adaptive Decision Engine"]
        IF[IsolationForest]
        LOF[LocalOutlierFactor]
        OCSVM[OneClassSVM]
        RF["RandomForest — Human Feedback Layer"]
    end

    subgraph UQ["Uncertainty Quantification"]
        EP[Epistemic Uncertainty]
        AL[Aleatoric Uncertainty]
        CI[Prediction Interval]
    end

    subgraph Decision["Decision and Routing"]
        HIGH["High Confidence → Auto-Execute"]
        LOW["Low Confidence → Human Queue"]
    end

    subgraph HiTL["Human-in-the-Loop"]
        QUEUE[Intervention Queue]
        RESOLVE["Approve / Override"]
        ADAPT[Model Adaptation]
    end

    subgraph Obs["Observability"]
        PROM[Prometheus Metrics]
        GRAF[Grafana Dashboards]
        LOG[Structured JSON Logs]
    end

    CSV --> MIS
    API_IN --> MIS
    MIS --> IMP --> MLCore
    MLCore --> UQ --> Decision
    LOW --> QUEUE --> RESOLVE --> ADAPT --> RF
    Decision --> PROM --> GRAF
```

---

## Features

### ✅ Implemented and Working

| Feature | Description |
|---------|-------------|
| **Multi-Algorithm Ensemble** | IsolationForest, LocalOutlierFactor, OneClassSVM with uncertainty-weighted voting |
| **Human-in-the-Loop** | Full cycle: flag low-confidence → queue → approve/override → model adapts |
| **Hard Mode Resilience** | Handles 20–30% missing or corrupted sensor fields via statistical imputation |
| **Human Feedback Adaptation** | RandomForest supervised layer retrained on human-corrected labels in real time |
| **Uncertainty Decomposition** | Epistemic (model) + Aleatoric (data) uncertainty per prediction |
| **Conformal Prediction Intervals** | Calibrated confidence intervals with finite-sample coverage guarantees |
| **Concept Drift Detection** | KS-test based distribution monitoring with automated retraining trigger |
| **Plain-English Explanations** | Every prediction includes a human-readable explanation |
| **Prometheus Metrics** | 13+ metrics: predictions, anomalies, interventions, drift, latency, missing data |
| **Structured JSON Logging** | Correlation-ID-tagged logs for full request tracing |
| **Kubernetes Health Probes** | `/health/live` and `/health/ready` endpoints ready for K8s |
| **Grafana Dashboards** | Pre-configured observability dashboards auto-provisioned on startup |
| **GitHub Actions CI** | Lint (ruff, oxlint), type-check (mypy), tests (pytest) |
| **React Live Dashboard** | Frontend with decision feed, intervention queue UI, and metrics panels |
| **Windows One-Click Setup** | `setup.ps1` bootstraps backend + frontend in separate windows |

### 🔭 Planned Enhancements

See the [Future Plans](#future-plans) section below.

| Feature | Target Phase |
|---------|-------------|
| Multimodal ingestion (images, text logs, time-series) | Phase 2 |
| SHAP explainability with frontend visualisation | Phase 2 |
| WebSocket real-time streaming | Phase 3 |
| Celery + Redis async task queue | Phase 3 |
| PostgreSQL persistent decision store | Phase 3 |
| Kubernetes manifests (dev / staging / prod) | Phase 4 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend API** | FastAPI 0.104+, Uvicorn, Pydantic v2 |
| **ML Engine** | scikit-learn (IsolationForest, LOF, OC-SVM, RandomForest), NumPy, Pandas |
| **Frontend** | React 18, Vite, vanilla CSS |
| **Observability** | Prometheus, Grafana, structlog |
| **Infrastructure** | Nginx |
| **Database (planned)** | PostgreSQL 15, Redis 7 |
| **CI/CD** | GitHub Actions |
| **Code Quality** | Ruff, MyPy, Black (Python) / oxlint (JS) |
| **Testing** | pytest, pytest-cov, httpx |

---

## Quick Start

### Prerequisites

- Python 3.11+ and Node.js 20+

---

### Option 1 — Local Development

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r ../requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

### Option 2 — Automated Windows Setup

```powershell
# Run from the project root
.\setup.ps1
```

---

## API Reference

### Base URL

```
http://localhost:8000
```

### Health and Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Basic health check |
| `GET` | `/health/live` | Kubernetes liveness probe |
| `GET` | `/health/ready` | Kubernetes readiness probe |
| `GET` | `/metrics` | Prometheus metrics (text/plain) |

### Data Streaming

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stream?count=5` | Batch of sensor readings with missing-data simulation |

### Decision Making

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/decision` | Submit sensor data, receive AI decision |

**Request body:**
```json
{
  "id": "evt_001",
  "timestamp": 1699999999.0,
  "temperature": 72.4,
  "pressure": 1.05,
  "vibration": 0.28,
  "source_reliable": true
}
```

**Response:**
```json
{
  "data_id": "evt_001",
  "prediction": "NORMAL_OPERATION",
  "confidence_score": 0.91,
  "explanation": "Ensemble (3 detectors): all sensor readings within normal historical bounds.",
  "requires_human": false
}
```

> When `requires_human` is `true`, the item is automatically added to the intervention queue.

### Human Intervention

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/interventions` | List all pending interventions |
| `POST` | `/api/interventions/{id}/resolve` | Approve or override a flagged decision |

**Resolve request:**
```json
{
  "approved": false,
  "new_prediction": "ANOMALY_DETECTED"
}
```

Human corrections are fed back into the RandomForest supervised layer immediately.

---

## Configuration

Copy `backend/.env.example` to `backend/.env` and customise:

```bash
# Core
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-secret-key-32-chars-min

# ML
MODEL__CONFIDENCE_THRESHOLD=0.70   # Below this → human intervention queue
MODEL__ENSEMBLE_ENABLED=true
MODEL__DRIFT_ENABLED=true
MODEL__FEEDBACK_BATCH_SIZE=5       # Retrain RF after N human corrections

# API
API__HOST=0.0.0.0
API__PORT=8000
API__WORKERS=4

# Monitoring
MONITORING__LOG_LEVEL=INFO
MONITORING__LOG_FORMAT=json
MONITORING__METRICS_ENABLED=true

# Storage
STORAGE__DATABASE_URL=postgresql://user:pass@localhost:5432/stamper_tslr
STORAGE__REDIS_URL=redis://localhost:6379/0
```

---

## Monitoring and Observability

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Requests by method / endpoint / status |
| `http_request_duration_seconds` | Histogram | API latency |
| `predictions_total` | Counter | Predictions by class and model version |
| `prediction_confidence` | Histogram | Confidence score distribution |
| `prediction_uncertainty_epistemic` | Histogram | Epistemic (model) uncertainty |
| `prediction_uncertainty_aleatoric` | Histogram | Aleatoric (data) uncertainty |
| `anomalies_detected_total` | Counter | Anomalies by detector |
| `interventions_total` | Counter | Human interventions by action |
| `intervention_queue_depth` | Gauge | Pending interventions count |
| `drift_detected_total` | Counter | Drift events by type |
| `drift_severity` | Gauge | Current drift severity |
| `model_retrains_total` | Counter | Retraining events |
| `missing_data_ratio` | Gauge | Missing field ratio per sensor |

### Grafana Dashboards

Auto-provisioned at http://localhost:3000:

- **Vigil Overview** — system health, latency, prediction rates, anomaly counts, intervention queue depth
- **ML Model Performance** — confidence calibration, uncertainty decomposition, drift trend
- **Business Metrics** — anomaly rate, human override rate, model adaptation velocity

### Structured Logging

```json
{
  "timestamp": "2026-08-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "http",
  "correlation_id": "a1b2c3d4",
  "event": "request_completed",
  "method": "POST",
  "path": "/api/decision",
  "status_code": 200,
  "duration_ms": 38.2
}
```

---

## Project Structure

```
STAMPER_TSLR/
├── backend/
│   ├── main.py              # API endpoints and lifespan management
│   ├── ml_engine.py         # Adaptive Decision Engine (ensemble + UQ + drift)
│   ├── data_generator.py    # CSV streaming with missing-data simulation
│   ├── config.py            # Pydantic Settings (env / YAML)
│   ├── monitoring.py        # Prometheus metrics, logging, health checks
│   ├── sensor_data.csv      # Historical training data
│   └── .env.example         # Environment variable template
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main dashboard component
│   │   ├── App.css          # Component styles
│   │   └── index.css        # Global design tokens
│   ├── nginx.conf           # Production static serving
│   └── package.json
│
├── monitoring/
│   ├── prometheus.yml       # Scrape config
│   └── grafana/
│       ├── datasources/     # Prometheus data source
│       └── dashboards/      # Pre-built dashboard JSON
│
├── tests/
│   ├── conftest.py          # Pytest fixtures and test client
│   ├── test_ml_engine.py    # ML engine unit tests
│   ├── test_api.py          # API integration tests
│   └── test_ml_features.py  # ML contract tests
│
├── requirements.txt         # Python dependencies
├── pytest.ini
├── setup.ps1                # Windows one-click dev setup
└── README.md
```

---

## Running Tests

```bash
cd backend
python -m venv venv && venv\Scripts\activate
pip install -r ../requirements.txt

# All tests with coverage
pytest ../tests/ -v --cov=. --cov-report=term-missing

# By category
pytest ../tests/ -m unit
pytest ../tests/ -m integration
pytest ../tests/ -m ml

# Frontend
cd ../frontend
npm run lint
npm run build
```

---

## Future Plans

The core MVP platform is complete and fully functional. The following enhancements are planned for the next development cycle:

### 🔷 Phase 2 — Multimodal Support and Explainability

- [ ] SHAP feature attribution (TreeSHAP + KernelSHAP) with frontend plots
- [ ] Extensible modality schema for text logs, thermal images, and time-series streams
- [ ] Cross-modal fusion (early / late / cross-attention)
- [ ] Counterfactual explanations ("what would change this decision?")

### 🔷 Phase 3 — Real-time Streaming and Persistence

- [ ] WebSocket endpoints for live decision streaming to the frontend
- [ ] Celery + Redis async task queue for heavy inference jobs
- [ ] PostgreSQL persistence for all decisions, interventions, and audit trail
- [ ] Model versioning and artifact storage (MinIO / S3)

### 🔷 Phase 4 — Production Hardening

- [ ] Kubernetes manifests with dev / staging / prod overlays via Kustomize
- [ ] Canary deployments with automated rollback
- [ ] KEDA-based autoscaling for Celery workers
- [ ] Load testing (Locust) and chaos engineering (LitmusChaos)
- [ ] Security hardening, RBAC, and compliance reporting

### 🔷 Phase 5 — Advanced ML

- [ ] Champion / challenger automated retraining pipeline
- [ ] Online learning for streaming concept drift adaptation
- [ ] Active learning — smart selection of highest-value human review cases
- [ ] Federated learning support for privacy-preserving distributed training

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for your changes
4. Run quality checks: `ruff check backend/`, `mypy backend/`, `pytest tests/`
5. Submit a Pull Request with a clear description

**Code Standards:**
- Python: Black formatting, Ruff linting, MyPy strict
- JavaScript: oxlint
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- Tests: maintain > 80% coverage for all new code

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/) — Core ML algorithms
- [FastAPI](https://fastapi.tiangolo.com/) — Modern async Python API framework
- [React](https://react.dev/) + [Vite](https://vitejs.dev/) — Frontend toolchain
- [Prometheus](https://prometheus.io/) + [Grafana](https://grafana.com/) — Observability stack
- [structlog](https://www.structlog.org/) — Structured logging

---

<div align="center">
  <strong>Built with ❤️ for reliable AI decision-making under uncertainty</strong><br>
  <em>"The system must quantify its confidence, explain why a decision was made, detect when its prediction may be unreliable, and request human intervention when necessary."</em>
</div>
