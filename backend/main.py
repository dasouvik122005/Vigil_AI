from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import time

from data_generator import DataStreamer
from ml_engine import AdaptiveDecisionEngine

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
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    vibration: Optional[float] = None
    source_reliable: bool = True

class DecisionResponse(BaseModel):
    data_id: str
    prediction: str
    confidence_score: float
    explanation: str
    requires_human: bool

# Initialize our new modules
streamer = DataStreamer(filename="sensor_data.csv")
ai_engine = AdaptiveDecisionEngine(historical_data_path="sensor_data.csv")

# Mock database for intervention queue and storage
intervention_queue = []
historical_decisions = {}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/stream", response_model=List[MultimodalData])
def get_data_stream(count: int = 5):
    """
    Streams realistic data from CSV.
    Implements Hard Mode: missing fields are handled dynamically.
    """
    raw_batch = streamer.get_next_batch(count=count)
    
    stream = []
    for raw in raw_batch:
        data = MultimodalData(
            id=f"evt_{int(time.time()*1000)}_{len(stream)}",
            timestamp=time.time(),
            temperature=raw['temperature'],
            pressure=raw['pressure'],
            vibration=raw['vibration'],
            source_reliable=not raw['is_corrupted']
        )
        stream.append(data)
    return stream

@app.post("/api/decision", response_model=DecisionResponse)
def make_decision(data: MultimodalData):
    """
    Uses the ML Engine for prediction, anomaly detection and confidence scoring.
    """
    data_dict = {
        'temperature': data.temperature,
        'pressure': data.pressure,
        'vibration': data.vibration
    }
    
    # Store raw data for future human feedback
    historical_decisions[data.id] = data_dict
    
    # 1. Run real ML inference
    result = ai_engine.predict(data_dict)
    
    prediction = "ANOMALY_DETECTED" if result["is_anomaly"] else "NORMAL_OPERATION"
    
    # 2. Hard Mode Check: Does the AI trust itself?
    requires_human = result["confidence"] < 0.70
    
    response = DecisionResponse(
        data_id=data.id,
        prediction=prediction,
        confidence_score=round(result["confidence"], 2),
        explanation=result["explanation"],
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
    """
    Resolves an item in the human-in-the-loop queue and
    FEEDS THE HUMAN CORRECTION BACK INTO THE AI ENGINE to adapt.
    """
    global intervention_queue
    
    for item in intervention_queue:
        if item["data_id"] == data_id:
            # 1. Remove from queue
            intervention_queue = [x for x in intervention_queue if x["data_id"] != data_id]
            
            # 2. Human Feedback Loop (Adaptation)
            raw_data = historical_decisions.get(data_id)
            if raw_data:
                # If approved, the original prediction was correct. 
                # If overridden, the new_prediction is the correct one.
                final_decision = item["prediction"] if approved else new_prediction
                is_anomaly_label = (final_decision == "ANOMALY_DETECTED")
                
                # Send back to ML Engine to learn!
                ai_engine.add_human_feedback(raw_data, is_anomaly_label)
                
            return {"status": "resolved", "data_id": data_id, "adapted": bool(raw_data)}
    
    raise HTTPException(status_code=404, detail="Item not found in intervention queue")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
