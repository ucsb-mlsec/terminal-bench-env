from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List
import uvicorn

# Request model for prediction
class PredictionRequest(BaseModel):
    features: List[float]

# Response model for prediction
class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

# Initialize FastAPI app
app = FastAPI()

# Global variable to store the model
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        with open("/home/user/model_api/model.pkl", "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Convert features to numpy array and reshape for single prediction
        features_array = np.array(request.features).reshape(1, -1)
        
        # Get prediction
        prediction = int(model.predict(features_array)[0])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features_array)[0]
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence
        )
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)