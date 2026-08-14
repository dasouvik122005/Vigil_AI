# STAMPER_TSLR: AI-Powered Adaptive Decision Intelligence Platform

Welcome to the STAMPER_TSLR project! This repository contains the MVP prototype for an Adaptive Decision Intelligence Platform built for Hackathon Track 1.

## Project Overview

The platform uses an AI model (currently utilizing Random Forests and Isolation Forests) to process data streams, assign confidence scores to its own predictions, and identify when it should not trust its own results. When confidence is low or anomalies are detected, the system triggers a human-in-the-loop fallback mechanism.

### Key Features
- **FastAPI Backend:** Provides robust and fast API endpoints for data streaming and ML model inference.
- **React + Vite Frontend:** A modern, dynamic web interface for monitoring AI decisions, confidence scores, and manual interventions.
- **Adaptive Machine Learning:** Evaluates data quality, scores predictions, and detects anomalies on the fly.
- **Resilience:** Designed to gracefully handle missing or corrupted data.

## Directory Structure
- `/backend`: Contains the Python/FastAPI backend, data generators, and ML engine logic.
- `/frontend`: Contains the Vite + React frontend application.

## Prerequisites
- Node.js (v16+)
- Python (3.9+)

## Setup Instructions

### 1. Backend Setup
Navigate to the `backend` directory, create a virtual environment, and install dependencies:

```bash
cd backend
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r ../requirements.txt
```

### 2. Frontend Setup
Navigate to the `frontend` directory and install NPM packages:

```bash
cd frontend
npm install
```

### 3. Running the Application
You can use the provided `setup.ps1` script on Windows to install dependencies and start both the backend and frontend simultaneously, or run them separately:

**Run Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Run Frontend:**
```bash
cd frontend
npm run dev
```

## Contributing
Please refer to the open issues and current `MEMORY.md` to see the roadmap and next steps for the project!
