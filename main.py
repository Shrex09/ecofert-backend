from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

# ----------------------------------------------------
# 1️⃣ Initialize FastAPI app
# ----------------------------------------------------
app = FastAPI(
    title="EcoFert Fertilizer Recommendation API",
    description="AI-based API that predicts the best organic fertilizer "
                "based on soil nutrient data.",
    version="2.0.0"
)

# ----------------------------------------------------
# 2️⃣ Enable CORS (so Flutter web/mobile can access)
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all for dev; restrict later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 3️⃣ Define Pydantic data model
# ----------------------------------------------------
class SoilData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    moisture: float
    temperature: float


# ----------------------------------------------------
# 4️⃣ Load ML Model
# ----------------------------------------------------
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Model file '{MODEL_PATH}' not found! Please upload a trained model first."
    )

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None


# ----------------------------------------------------
# 5️⃣ Define prediction endpoint
# ----------------------------------------------------
@app.post("/predict")
def predict(data: SoilData):
    """
    Predict the best fertilizer based on NPK, pH, moisture, and temperature.
    Returns a fertilizer name from the trained ML model.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    # Convert input data to NumPy array
    X = np.array([[data.nitrogen, data.phosphorus, data.potassium,
                   data.ph, data.moisture, data.temperature]])

    try:
        # Run model prediction
        prediction = model.predict(X)[0]
        return {"recommended_fertilizer": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ----------------------------------------------------
# 6️⃣ Root endpoint for sanity check
# ----------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "🌱 EcoFert API is running successfully!",
        "status": "OK",
        "endpoints": ["/predict"]
    }
