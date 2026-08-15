# setup.ps1
# Script to install dependencies and start both frontend and backend for Vigil AI

Write-Host "Setting up Vigil AI Project..." -ForegroundColor Green

# 1. Setup Backend
Write-Host "Setting up Python Backend..." -ForegroundColor Cyan
Set-Location -Path .\backend

if (-Not (Test-Path -Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

Write-Host "Activating virtual environment and installing dependencies..."
& .\venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt

# Start backend in a new background job or window
Write-Host "Starting FastAPI Backend on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '.\venv\Scripts\Activate.ps1'; uvicorn main:app --reload --port 8000"

Set-Location -Path ..

# 2. Setup Frontend
Write-Host "Setting up React Frontend..." -ForegroundColor Cyan
Set-Location -Path .\frontend

Write-Host "Installing NPM dependencies..."
npm install

# Start frontend
Write-Host "Starting Vite Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev"

Set-Location -Path ..

Write-Host "Setup complete! The backend and frontend are now running in separate windows." -ForegroundColor Green
