# train_model.py — EcoFert Kaggle ML Trainer 🌾
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🌱 Starting EcoFert (Kaggle) Model Training...")

# ---------------------------
# 1️⃣ Load Kaggle Dataset
# ---------------------------
try:
    df = pd.read_csv("Fertilizer_Prediction.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    raise Exception("❌ Dataset file 'Fertilizer Prediction.csv' not found. Please place it in this folder.")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")
print("📊 Columns found:", df.columns.tolist())

# ---------------------------
# 2️⃣ Select Relevant Columns
# ---------------------------
required_cols = [
    "Nitrogen", "Phosphorous", "Potassium",
    "Moisture", "Temparature", "Crop_Type", "Fertilizer_Name"
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise Exception(f"❌ Missing required columns in dataset: {missing}")

df = df[required_cols].dropna()
print(f"✅ Using {len(df)} valid rows.")

# Rename for consistency
df.rename(columns={
    "Nitrogen": "nitrogen",
    "Phosphorous": "phosphorus",
    "Potassium": "potassium",
    "Moisture": "moisture",
    "Temparature": "temperature",
    "Crop_Type": "crop",
    "Fertilizer_Name": "fertilizer"
}, inplace=True)

# ---------------------------
# 3️⃣ Encode Target Labels
# ---------------------------
fertilizer_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

df["fertilizer_label"] = fertilizer_encoder.fit_transform(df["fertilizer"])
df["crop_label"] = crop_encoder.fit_transform(df["crop"])

# ---------------------------
# 4️⃣ Split Features and Targets
# ---------------------------
X = df[["nitrogen", "phosphorus", "potassium", "moisture", "temperature"]]
y_fertilizer = df["fertilizer_label"]
y_crop = df["crop_label"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y_fertilizer, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_crop, test_size=0.2, random_state=42)

# ---------------------------
# 5️⃣ Train Random Forest Models
# ---------------------------
print("🚜 Training Random Forest models...")

fertilizer_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
fertilizer_model.fit(X_train_f, y_train_f)

crop_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
crop_model.fit(X_train_c, y_train_c)

# ---------------------------
# 6️⃣ Evaluate Performance
# ---------------------------
fert_acc = accuracy_score(y_test_f, fertilizer_model.predict(X_test_f))
crop_acc = accuracy_score(y_test_c, crop_model.predict(X_test_c))

print(f"🌿 Fertilizer Model Accuracy: {fert_acc:.2%}")
print(f"🌾 Crop Model Accuracy: {crop_acc:.2%}")

print("\n📈 Fertilizer Classification Report:")
print(classification_report(y_test_f, fertilizer_model.predict(X_test_f), zero_division=0))

print("\n📈 Crop Classification Report:")
print(classification_report(y_test_c, crop_model.predict(X_test_c), zero_division=0))

# ---------------------------
# 7️⃣ Save Models and Encoders
# ---------------------------
joblib.dump(fertilizer_model, "fertilizer_model.pkl")
joblib.dump(crop_model, "crop_model.pkl")
joblib.dump(fertilizer_encoder, "fertilizer_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")

print("\n✅ Model and encoder files saved successfully:")
print(" - fertilizer_model.pkl")
print(" - crop_model.pkl")
print(" - fertilizer_encoder.pkl")
print(" - crop_encoder.pkl")

# ---------------------------
# 8️⃣ Example Test Prediction
# ---------------------------
sample = [[50, 40, 45, 55, 26]]  # N, P, K, Moisture, Temperature
fert_pred = fertilizer_encoder.inverse_transform(fertilizer_model.predict(sample))[0]
crop_pred = crop_encoder.inverse_transform(crop_model.predict(sample))[0]

print("\n🌱 Example Prediction:")
print(f"   Fertilizer → {fert_pred}")
print(f"   Crop       → {crop_pred}")

print("\n✅ EcoFert Kaggle ML training completed successfully.")
