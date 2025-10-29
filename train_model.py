# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------
# 1. Load or create sample soil dataset
# ---------------------------
data = {
    'nitrogen': [40, 60, 80, 30, 90, 55, 45, 70, 85, 20],
    'phosphorus': [20, 40, 60, 15, 70, 50, 25, 55, 65, 10],
    'potassium': [30, 50, 70, 25, 75, 55, 40, 60, 80, 15],
    'ph': [6.2, 7.0, 6.8, 5.8, 7.5, 6.5, 6.0, 6.9, 7.2, 5.5],
    'moisture': [45, 60, 70, 35, 80, 50, 40, 65, 75, 30],
    'temperature': [26, 29, 31, 24, 32, 28, 27, 30, 33, 23],
    'fertilizer': ['Compost', 'Neem Cake', 'Organic Mix', 'Cow Dung',
                   'Organic Mix', 'Neem Cake', 'Compost', 'Neem Cake', 'Organic Mix', 'Cow Dung']
}

df = pd.DataFrame(data)

# ---------------------------
# 2. Split features and target
# ---------------------------
X = df[['nitrogen', 'phosphorus', 'potassium', 'ph', 'moisture', 'temperature']]
y = df['fertilizer']

# ---------------------------
# 3. Train Random Forest model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------------------
# 4. Save trained model
# ---------------------------
joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as model.pkl")
