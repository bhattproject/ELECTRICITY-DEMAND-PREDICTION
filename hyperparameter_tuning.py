
"""
Hyperparameter Tuning for Electricity Demand Prediction

This script performs extensive hyperparameter tuning for multiple
machine learning models using GridSearchCV and RandomizedSearchCV.
It supports:
- Random Forest
- Gradient Boosting
- XGBoost
- Placeholder for LSTM / Deep Learning models
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
DATA_DIR = "data_splits"
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
OUTPUT_DIR = "tuned_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CV_SPLITS = 5  
def evaluate_model(model, X_val, y_val):
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return mae, rmse


print("Tuning Random Forest Regressor...")
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
rf_mae, rf_rmse = evaluate_model(best_rf, X_val, y_val)
joblib.dump(best_rf, os.path.join(OUTPUT_DIR, "best_rf_model.pkl"))
print(f"Best RF Params: {rf_grid.best_params_}")
print(f"Validation MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}")

print("\nTuning Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(random_state=42)
gb_param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 1.0]
}
gb_grid = GridSearchCV(gb_model, gb_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
gb_grid.fit(X_train, y_train)

best_gb = gb_grid.best_estimator_
gb_mae, gb_rmse = evaluate_model(best_gb, X_val, y_val)
joblib.dump(best_gb, os.path.join(OUTPUT_DIR, "best_gb_model.pkl"))
print(f"Best GB Params: {gb_grid.best_params_}")
print(f"Validation MAE: {gb_mae:.4f}, RMSE: {gb_rmse:.4f}")

print("\nTuning XGBoost Regressor...")
xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')
xgb_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_
xgb_mae, xgb_rmse = evaluate_model(best_xgb, X_val, y_val)
joblib.dump(best_xgb, os.path.join(OUTPUT_DIR, "best_xgb_model.pkl"))
print(f"Best XGB Params: {xgb_grid.best_params_}")
print(f"Validation MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}")

print("\nLSTM / Deep Learning models can be tuned here using Keras Tuner or Optuna.")

print("\nEvaluating best models on Test Set:")
models = {'RandomForest': best_rf, 'GradientBoosting': best_gb, 'XGBoost': best_xgb}
for name, model in models.items():
    mae, rmse = evaluate_model(model, X_test, y_test)
    print(f"{name} Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")
