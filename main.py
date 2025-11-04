# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

# ---------------------------
# 🌱 App Initialization
# ---------------------------
app = FastAPI(title="EcoFert Crop & Fertilizer Recommendation API")

# Allow requests from Flutter & Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 📦 Load Trained Models
# ---------------------------
try:
    fertilizer_model = joblib.load("fertilizer_model.pkl")
    crop_model = joblib.load("crop_model.pkl")
    fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")
    crop_encoder = joblib.load("crop_encoder.pkl")
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    fertilizer_model = None
    crop_model = None
    fertilizer_encoder = None
    crop_encoder = None


# ---------------------------
# 🌾 Define Input Schema with Validations
# ---------------------------
class SoilData(BaseModel):
    nitrogen: float = Field(..., ge=0, le=150, description="Nitrogen content (0–150)")
    phosphorus: float = Field(..., ge=0, le=150, description="Phosphorus content (0–150)")
    potassium: float = Field(..., ge=0, le=200, description="Potassium content (0–200)")
    ph: float = Field(..., ge=3.5, le=9.0, description="Soil pH value (3.5–9.0)")
    moisture: float = Field(..., ge=0, le=100, description="Moisture percentage (0–100)")
    temperature: float = Field(..., ge=5, le=50, description="Temperature °C (5–50)")


# ---------------------------
# 🌿 Root Route
# ---------------------------
@app.get("/")
def root():
    return {"status": "OK", "message": "EcoFert API running 🌿"}


# ---------------------------
# 🔮 Prediction Route
# ---------------------------
@app.post("/predict")
def predict(data: SoilData):
    # Check models
    if None in [fertilizer_model, crop_model, fertilizer_encoder, crop_encoder]:
        raise HTTPException(status_code=500, detail="Models not loaded properly")

    try:
        # Convert input data to numpy
        X = np.array([[data.nitrogen, data.phosphorus, data.potassium,
                       data.moisture, data.temperature]])

        # Run predictions
        fert_pred = fertilizer_encoder.inverse_transform(
            fertilizer_model.predict(X)
        )[0]
        crop_pred = crop_encoder.inverse_transform(
            crop_model.predict(X)
        )[0]

        print(f"🌾 Predicted → Crop: {crop_pred}, Fertilizer: {fert_pred}")

        return {
            "recommended_crop": crop_pred,
            "recommended_fertilizer": fert_pred
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# ---------------------------
# 🚦 Health Check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": fertilizer_model is not None}

