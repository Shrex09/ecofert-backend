from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EcoFert Fertilizer Recommendation API")

# Allow requests from your Flutter app (localhost, web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Data model for incoming JSON
# ---------------------------
class SoilData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    moisture: float
    temperature: float


# ---------------------------
# Dummy or ML-based prediction
# ---------------------------
# If you have a trained model, uncomment:
# model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: SoilData):
    # Convert data to numpy array if needed
    X = np.array([[data.nitrogen, data.phosphorus, data.potassium, data.ph, data.moisture, data.temperature]])

    # ---- Option 1: Use a trained ML model ----
    # y_pred = model.predict(X)[0]
    # fertilizer = y_pred

    # ---- Option 2: Simple rules for now ----
    if data.nitrogen < 50:
        fertilizer = "Compost"
    elif data.ph < 6.5:
        fertilizer = "Neem Cake"
    elif data.moisture < 40:
        fertilizer = "Cow Dung"
    else:
        fertilizer = "Organic Mix"

    # Return as JSON
    return {"recommended_fertilizer": fertilizer}
