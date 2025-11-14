"""
Electricity Demand Forecasting Model
Author: Your Name
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# ==========================
# 1. LOAD DATASET
# ==========================

DATA_FILE = "Electricity_Dataset.csv"
assert os.path.exists(DATA_FILE), f"Dataset  not found: {DATA_FILE}"

print("its the dataset loading ...")
df = pd.read_csv(DATA_FILE)

# Ensure column names
if "DateTime" not in df.columns or "Demand" not in df.columns:
    raise ValueError("Dataset must contain 'DateTime' and 'Demand' columns")

# Convert to datetime
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime")
df = df.reset_index(drop=True)

# ==========================
# 2. FEATURE ENGINEERING
# ==========================

print("features are adding...")

df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Month"] = df["DateTime"].dt.month
df["DayOfWeek"] = df["DateTime"].dt.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

# Lag & Rolling features
df["Lag1"] = df["Demand"].shift(1)
df["Lag24"] = df["Demand"].shift(24)
df["Rolling24"] = df["Demand"].shift(1).rolling(24).mean()
df["Rolling7"] = df["Demand"].shift(1).rolling(7).mean()

df = df.dropna().reset_index(drop=True)

FEATURES = ["Hour", "Day", "Month", "DayOfWeek", "IsWeekend", "Lag1", "Lag24", "Rolling24", "Rolling7"]

X = df[FEATURES]
y = df["Demand"]

# ==========================
# 3. TRAIN-TEST SPLIT
# ==========================

print("Splitting dataset...")
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ==========================
# 4. TRAINING MODEL
# ==========================

print("Model training started.....")
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# ==========================
# 5. EVALUATION
# ==========================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n===== MODEL PERFORMANCE =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# ==========================
# 6. SAVE MODEL
# ==========================

MODEL_FILE = "electricity_model.pkl"
joblib.dump(model, MODEL_FILE)
print(f"Model saved as {MODEL_FILE}")

# ==========================
# 7. PREDICTION FUNCTION
# ==========================

def predict_demand(timestamp):
    """Predict electricity demand for a given timestamp."""
    ts = pd.to_datetime(timestamp)

    last_row = df.iloc[-1]

    temp = pd.DataFrame({
        "Hour": [ts.hour],
        "Day": [ts.day],
        "Month": [ts.month],
        "DayOfWeek": [ts.dayofweek],
        "IsWeekend": [int(ts.dayofweek >= 5)],
        "Lag1": [last_row["Demand"]],
        "Lag24": [df.iloc[-24]["Demand"]],
        "Rolling24": [df["Demand"].iloc[-24:].mean()],
        "Rolling7": [df["Demand"].iloc[-7:].mean()],
    })

    loaded_model = joblib.load(MODEL_FILE)
    pred = loaded_model.predict(temp)[0]
    return pred

# ==========================
# 8. TEST PREDICTION
# ==========================

# Manual test example
# predicted_value = predict_demand("2024-10-01 15:00:00")

# print(predicted_value)

# Manual test example
# predicted_value = predict_demand("2024-04-01 20:00:00")

# print(predicted_value)

