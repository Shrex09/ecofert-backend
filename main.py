# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="EcoFert Crop & Fertilizer API")

# Allow requests from Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load trained models
# ---------------------------
try:
    fertilizer_model = joblib.load("fertilizer_model.pkl")
    crop_model = joblib.load("crop_model.pkl")
    fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")
    crop_encoder = joblib.load("crop_encoder.pkl")
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"⚠ Error loading models: {e}")
    fertilizer_model = None
    crop_model = None


# ---------------------------
# Input schema
# ---------------------------
class SoilData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    moisture: float
    temperature: float


# ---------------------------
# Predict endpoint (returns both crop + fertilizer)
# ---------------------------
@app.post("/predict")
def predict(data: SoilData):
    if fertilizer_model is None or crop_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Prepare input for model
    features = np.array([[data.nitrogen, data.phosphorus, data.potassium,
                          data.moisture, data.temperature]])

    fert_pred = fertilizer_encoder.inverse_transform(fertilizer_model.predict(features))[0]
    crop_pred = crop_encoder.inverse_transform(crop_model.predict(features))[0]

    print(f"🌾 Prediction: Crop={crop_pred}, Fertilizer={fert_pred}")

    return {
        "recommended_crop": crop_pred,
        "recommended_fertilizer": fert_pred
    }


@app.get("/")
def root():
    return {"status": "OK", "message": "EcoFert API running 🌿"}