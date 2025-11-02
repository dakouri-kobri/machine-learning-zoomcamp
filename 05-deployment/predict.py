import pickle
from typing import Any, Dict

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="churn-prediction")


# Load saved model from a file
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer: Dict[str, Any]) -> float:
    """
    Predict churn probability for one customer dict.
    Ensures correct shape and returns a plain Python float.
    """
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

# Prediction on a single case
@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)

    return {
        "churn_probability": prob,
        "churn": prob >= 0.5
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)