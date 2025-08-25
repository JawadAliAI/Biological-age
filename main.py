from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pickle
import numpy as np
import time
import uvicorn

# ---------------------------------------------------
# Initialize FastAPI app
# ---------------------------------------------------
app = FastAPI(
    title="Phenoage Prediction API",
    description="Biomarker-based biological age prediction using Aira Mandatory File"
)

# ---------------------------------------------------
# Load the trained model
# ---------------------------------------------------
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ---------------------------------------------------
# Load the scaler
# ---------------------------------------------------
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# ---------------------------------------------------
# Pydantic models
# ---------------------------------------------------
class BiomarkerInput(BaseModel):
    age_years: float = Field(..., ge=0, le=120)
    albumin_gl: float = Field(..., ge=20, le=60)
    creatinine_umoll: float = Field(..., ge=40, le=400)
    glucose_mmoll: float = Field(..., ge=3, le=30)
    crp_mgdl: float = Field(..., ge=0, le=50)
    lymphocyte_percent: float = Field(..., ge=5, le=60)
    mcv_fl: float = Field(..., ge=70, le=110)
    rdw_percent: float = Field(..., ge=10, le=25)
    alkp_ul: float = Field(..., ge=30, le=300)
    wbc_10_9l: float = Field(..., ge=2, le=20)
    chronic_age: float = Field(..., ge=0, le=120)

class PredictionResponse(BaseModel):
    predicted_biological_age: float
    status: str
    model_type: str
    processing_time: str

# ---------------------------------------------------
# Health check
# ---------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_biological_age(biomarkers: BiomarkerInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        start_time = time.time()

        # Prepare input in correct order (MUST match training order!)
        input_data = np.array([[
            biomarkers.albumin_gl,
            biomarkers.creatinine_umoll,
            biomarkers.glucose_mmoll,
            biomarkers.crp_mgdl,
            biomarkers.lymphocyte_percent,
            biomarkers.mcv_fl,
            biomarkers.rdw_percent,
            biomarkers.alkp_ul,
            biomarkers.wbc_10_9l,
            biomarkers.chronic_age
            # ⚠️ if age_years was used in training, add it back here!
        ]])

        # Apply scaling
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        processing_time = time.time() - start_time

        return PredictionResponse(
            predicted_biological_age=float(prediction),
            status="Success",
            model_type=type(model).__name__,   # Auto-detect model name
            processing_time=f"{processing_time:.3f} seconds"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ---------------------------------------------------
# Serve the main HTML page
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

# ---------------------------------------------------
# Run app
# ---------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
