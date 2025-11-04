# train_models.py (EcoFert Organic Version)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

print("🌱 Starting EcoFert Organic Model Training...")

# ---------------------------
# 1. Load the dataset
# ---------------------------
df = pd.read_csv("EcoFert_Organic.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

print("📊 Columns:", df.columns.tolist())

# ---------------------------
# 2. Keep only relevant columns
# ---------------------------
required_cols = [
    "nitrogen", "phosphorus", "potassium",
    "ph", "moisture", "temperature",
    "recommended_fertilizer", "recommended_crop"
]
df = df[required_cols].dropna()

print(f"✅ Loaded {len(df)} valid rows.")

# ---------------------------
# 3. Encode categorical labels
# ---------------------------
fertilizer_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

df["fertilizer_label"] = fertilizer_encoder.fit_transform(df["recommended_fertilizer"])
df["crop_label"] = crop_encoder.fit_transform(df["recommended_crop"])

# ---------------------------
# 4. Split data for two models
# ---------------------------
X = df[["nitrogen", "phosphorus", "potassium", "ph", "moisture", "temperature"]]
y_fert = df["fertilizer_label"]
y_crop = df["crop_label"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y_fert, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_crop, test_size=0.2, random_state=42)

# ---------------------------
# 5. Train two Random Forests
# ---------------------------
fertilizer_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
fertilizer_model.fit(X_train_f, y_train_f)

crop_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
crop_model.fit(X_train_c, y_train_c)

# ---------------------------
# 6. Evaluate models
# ---------------------------
fert_acc = accuracy_score(y_test_f, fertilizer_model.predict(X_test_f))
crop_acc = accuracy_score(y_test_c, crop_model.predict(X_test_c))

print(f"🌿 Fertilizer Model Accuracy: {fert_acc:.2%}")
print(f"🌾 Crop Model Accuracy: {crop_acc:.2%}")

# ---------------------------
# 7. Save models and encoders
# ---------------------------
joblib.dump(fertilizer_model, "fertilizer_model.pkl")
joblib.dump(crop_model, "crop_model.pkl")
joblib.dump(fertilizer_encoder, "fertilizer_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")

print("\n✅ Models and encoders saved successfully:")
print(" - fertilizer_model.pkl")
print(" - crop_model.pkl")
print(" - fertilizer_encoder.pkl")
print(" - crop_encoder.pkl")

# ---------------------------
# 8. Example prediction
# ---------------------------
sample = [[60, 40, 55, 6.8, 65, 28]]
fert_pred = fertilizer_encoder.inverse_transform(fertilizer_model.predict(sample))[0]
crop_pred = crop_encoder.inverse_transform(crop_model.predict(sample))[0]

print("\n🌱 Example Prediction:")
print(f"   Fertilizer → {fert_pred}")
print(f"   Crop       → {crop_pred}")
