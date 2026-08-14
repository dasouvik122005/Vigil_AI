from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
from typing import Optional, List, Dict
import time

app = FastAPI(title="Adaptive Decision Intelligence API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MultimodalData(BaseModel):
    id: str
    timestamp: float
    text_input: Optional[str] = None
    sensor_reading: Optional[float] = None
    image_features: Optional[List[float]] = None
    source_reliable: bool = True

class DecisionResponse(BaseModel):
    data_id: str
    prediction: str
    confidence_score: float
    explanation: str
    requires_human: bool

# Mock database for intervention queue
intervention_queue = []

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/stream", response_model=List[MultimodalData])
def get_data_stream(count: int = 5):
    """
    Simulates a stream of multimodal data.
    Implements Hard Mode: 20-30% of data is missing or corrupted.
    """
    stream = []
    for _ in range(count):
        is_corrupted = random.random() < 0.25 # 25% chance of missing data
        
        data = MultimodalData(
            id=f"evt_{random.randint(1000, 9999)}",
            timestamp=time.time(),
            text_input="Sample observation from edge device" if not is_corrupted else None,
            sensor_reading=round(random.uniform(20.0, 100.0), 2) if not is_corrupted else None,
            image_features=[random.random() for _ in range(3)] if not is_corrupted else None,
            source_reliable=not is_corrupted
        )
        stream.append(data)
    return stream

@app.post("/api/decision", response_model=DecisionResponse)
def make_decision(data: MultimodalData):
    """
    Simulates AI decision making, anomaly detection and confidence scoring.
    """
    # Calculate simulated confidence based on data completeness
    missing_fields = 0
    if data.text_input is None: missing_fields += 1
    if data.sensor_reading is None: missing_fields += 1
    if data.image_features is None: missing_fields += 1
    
    # Base confidence is high if all data is present
    base_confidence = 0.95
    confidence = base_confidence - (missing_fields * 0.3)
    
    # Add some randomness to confidence
    confidence = max(0.1, min(0.99, confidence + random.uniform(-0.1, 0.1)))
    
    # Anomaly detection logic (mocked)
    is_anomaly = False
    if data.sensor_reading and data.sensor_reading > 85.0:
        is_anomaly = True
        
    prediction = "ANOMALY_DETECTED" if is_anomaly else "NORMAL_OPERATION"
    
    explanation = f"Decision made based on {3 - missing_fields}/3 data modalities. "
    if is_anomaly:
        explanation += "Sensor reading exceeded normal thresholds."
    elif missing_fields > 0:
        explanation += "Some data streams are unavailable, relying on partial information."
        
    requires_human = confidence < 0.70
    
    response = DecisionResponse(
        data_id=data.id,
        prediction=prediction,
        confidence_score=round(confidence, 2),
        explanation=explanation,
        requires_human=requires_human
    )
    
    if requires_human:
        intervention_queue.append(response.dict())
        
    return response

@app.get("/api/interventions")
def get_intervention_queue():
    """Returns decisions that require human review."""
    return intervention_queue

@app.post("/api/interventions/{data_id}/resolve")
def resolve_intervention(data_id: str, approved: bool, new_prediction: Optional[str] = None):
    """Resolves an item in the human-in-the-loop queue."""
    global intervention_queue
    for item in intervention_queue:
        if item["data_id"] == data_id:
            intervention_queue = [x for x in intervention_queue if x["data_id"] != data_id]
            return {"status": "resolved", "data_id": data_id, "approved": approved, "updated": new_prediction}
    
    raise HTTPException(status_code=404, detail="Item not found in intervention queue")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
