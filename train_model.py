# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# ---------------------------
# 1. Load the real dataset
# ---------------------------
df = pd.read_csv("Fertilizer_Prediction.csv")

# Clean column names (some may have spaces or capitalization differences)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Show available columns (optional debug)
print("📊 Columns:", df.columns.tolist())

# ---------------------------
# 2. Keep only relevant columns
# ---------------------------
selected_cols = ["Nitrogen", "Phosphorous", "Potassium", "Moisture", "Temparature", "Crop_Type", "Fertilizer_Name"]
df = df[selected_cols].dropna()

# Rename for consistency with the app
df.rename(columns={
    "Phosphorous": "phosphorus",
    "Potassium": "potassium",
    "Nitrogen": "nitrogen",
    "Moisture": "moisture",
    "Temparature": "temperature",
    "Crop_Type": "crop",
    "Fertilizer_Name": "fertilizer"
}, inplace=True)

print(f"✅ Using {len(df)} rows with features: nitrogen, phosphorus, potassium, moisture, temperature")

# ---------------------------
# 3. Encode categorical targets
# ---------------------------
fertilizer_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

df["fertilizer_label"] = fertilizer_encoder.fit_transform(df["fertilizer"])
df["crop_label"] = crop_encoder.fit_transform(df["crop"])

# ---------------------------
# 4. Split features and targets
# ---------------------------
X = df[["nitrogen", "phosphorus", "potassium", "moisture", "temperature"]]

y_fertilizer = df["fertilizer_label"]
y_crop = df["crop_label"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y_fertilizer, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_crop, test_size=0.2, random_state=42)

# ---------------------------
# 5. Train Random Forest models
# ---------------------------
fertilizer_model = RandomForestClassifier(n_estimators=200, random_state=42)
fertilizer_model.fit(X_train_f, y_train_f)

crop_model = RandomForestClassifier(n_estimators=200, random_state=42)
crop_model.fit(X_train_c, y_train_c)

# ---------------------------
# 6. Evaluate accuracy
# ---------------------------
fertilizer_acc = accuracy_score(y_test_f, fertilizer_model.predict(X_test_f))
crop_acc = accuracy_score(y_test_c, crop_model.predict(X_test_c))

print(f"🌿 Fertilizer model accuracy: {fertilizer_acc:.2%}")
print(f"🌾 Crop model accuracy: {crop_acc:.2%}")

# ---------------------------
# 7. Save both models and encoders
# ---------------------------
joblib.dump(fertilizer_model, "fertilizer_model.pkl")
joblib.dump(crop_model, "crop_model.pkl")
joblib.dump(fertilizer_encoder, "fertilizer_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")

print("✅ Models saved successfully:")
print(" - fertilizer_model.pkl")
print(" - crop_model.pkl")
print(" - fertilizer_encoder.pkl")
print(" - crop_encoder.pkl")

# ---------------------------
# 8. Example test prediction
# ---------------------------
sample = [[60, 40, 55, 65, 28]]
fert_pred = fertilizer_encoder.inverse_transform(fertilizer_model.predict(sample))[0]
crop_pred = crop_encoder.inverse_transform(crop_model.predict(sample))[0]

print(f"🌱 Example Prediction:\n  Crop: {crop_pred}\n  Fertilizer: {fert_pred}")