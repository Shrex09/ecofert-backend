# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# ---------------------------
# 🌾 App Initialization
# ---------------------------
app = FastAPI(title="EcoFert Crop & Fertilizer Prediction API")

# Allow cross-origin requests (for Flutter frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 📦 Load Trained ML Models
# ---------------------------
try:
    fertilizer_model = joblib.load("fertilizer_model.pkl")
    crop_model = joblib.load("crop_model.pkl")
    fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")
    crop_encoder = joblib.load("crop_encoder.pkl")
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    fertilizer_model = crop_model = None


# ---------------------------
# 🌱 Define Input Schema
# ---------------------------
class SoilData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    moisture: float
    temperature: float


# ---------------------------
# 🔮 Predict Endpoint
# ---------------------------
@app.post("/predict")
def predict(data: SoilData):
    if fertilizer_model is None or crop_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly")

    try:
        # Match model training feature order (6 features)
        X = np.array([[data.nitrogen, data.phosphorus, data.potassium,
                       data.ph, data.moisture, data.temperature]])

        fert_pred = fertilizer_encoder.inverse_transform(fertilizer_model.predict(X))[0]
        crop_pred = crop_encoder.inverse_transform(crop_model.predict(X))[0]

        print(f"🌾 Prediction Successful → Crop: {crop_pred}, Fertilizer: {fert_pred}")

        return {
            "recommended_crop": crop_pred,
            "recommended_fertilizer": fert_pred
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# ---------------------------
# 🌍 Health + Root Routes
# ---------------------------
@app.get("/")
def root():
    return {"status": "OK", "message": "EcoFert API running using Kaggle dataset 🌿"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": fertilizer_model is not None and crop_model is not None
    }
