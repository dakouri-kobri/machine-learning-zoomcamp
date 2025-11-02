# Package imports
import pickle
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Request
class Features(BaseModel):
    lead_source: str
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)

# Response 
class ScoreResponse(BaseModel):
    score: float  # probability of conversion

# App & model 

app = FastAPI(title="pipeline_v1 prediction")

with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

def predict_single(x: Dict[str, Any]) -> float:
    try:
        proba = pipeline.predict_proba([x])[0, 1]
        return float(proba)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# Return ONLY the score for the exercise
@app.post("/predict-score", response_model=ScoreResponse)
def score(payload: Features) -> ScoreResponse:
    return ScoreResponse(score=predict_single(payload.model_dump()))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9696)