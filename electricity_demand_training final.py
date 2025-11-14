"""
Electricity Demand Prediction - Full Pipeline (single source file)
Author: bhatt
Date: 10 Nov 2025

Notes:
- Place this file in the same folder as 'Electricity_Dataset.csv'.
- The script trains models, compares them, and saves the best model and scaler to 'models/'.
- Prediction helper supports:
    * forecasting for timestamps already present in the dataset (uses precomputed features)
    * forecasting for a future timestamp (builds features from most recent history)
"""

import os
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# ----------------------
# Configuration
# ----------------------
DATA_FILE = "Electricity_Dataset.csv"   # dataset file name (keep as provided)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Optional libraries (use if installed)
try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ----------------------
# # load data
# ----------------------
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # detect datetime column
    datetime_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower():
            datetime_col = c
            break
    if datetime_col is None:
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > 0.5 * len(parsed):
                    datetime_col = c
                    break
            except Exception:
                continue
    if datetime_col is None:
        raise ValueError("No datetime column detected. Add a column with dates/times.")

    # detect target numeric column (prefer common names)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found to use as target.")

    preferred = ["demand", "load", "consumption", "power", "usage", "value"]
    target_col = None
    for name in preferred:
        for c in numeric_cols:
            if name in c.lower():
                target_col = c
                break
        if target_col:
            break
    if target_col is None:
        # choose numeric column with max variance
        variances = {c: df[c].var() for c in numeric_cols}
        target_col = max(variances, key=variances.get)

    # keep only timestamp and target
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)
    df = df[[datetime_col, target_col]].rename(columns={datetime_col: "timestamp", target_col: "demand"})

    # resample to hourly and interpolate if needed (common for electricity)
    try:
        df = df.set_index("timestamp").resample("H").mean().interpolate().reset_index()
    except Exception:
        # if resampling fails, keep original ordering
        df = df.reset_index(drop=True)

    return df


# ----------------------
# # create features
# ----------------------
def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # lag features (hours)
    lags = [1, 24, 48, 72, 168]  # 1h, 1d, 2d, 3d, 7d
    for lag in lags:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # rolling statistics
    roll_windows = [24, 48, 72, 168]
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df["demand"].shift(1).rolling(window=w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["demand"].shift(1).rolling(window=w, min_periods=1).std().fillna(0)

    # simple time index trend
    df["time_idx"] = np.arange(len(df))

    # drop NaNs created by shifts
    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------
# # train model
# ----------------------
def train_and_tune(df: pd.DataFrame, model_dir: str = MODEL_DIR) -> Dict:
    df_feat = create_advanced_features(df)
    feature_cols = [c for c in df_feat.columns if c not in ("timestamp", "demand")]
    X = df_feat[feature_cols]
    y = df_feat["demand"]

    # time-based split: 80% train, 20% test
    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx].reset_index(drop=True)
    test_df = df_feat.iloc[split_idx:].reset_index(drop=True)

    X_train = train_df[feature_cols].copy()
    y_train = train_df["demand"].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["demand"].copy()

    # scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    # prepare candidate models and parameter distributions
    candidates = {}

    # RandomForest
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    candidates["RandomForest"] = {
        "model": rf,
        "param_dist": {
            "n_estimators": [200, 400, 800],
            "max_depth": [8, 16, 24, None],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
    }

    # LightGBM 
    if LGB_AVAILABLE:
        lgb = LGBMRegressor(random_state=42, n_jobs=-1)
        candidates["LightGBM"] = {
            "model": lgb,
            "param_dist": {
                "n_estimators": [600, 1000],
                "learning_rate": [0.01, 0.03, 0.05],
                "num_leaves": [24, 40, 80],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "max_depth": [-1, 8, 16],
            }
        }

    # XGBoost 
    if XGB_AVAILABLE:
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
        candidates["XGBoost"] = {
            "model": xgb_model,
            "param_dist": {
                "n_estimators": [600, 1000],
                "learning_rate": [0.01, 0.03, 0.05],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }
        }

    results = {}
    tscv = TimeSeriesSplit(n_splits=5)
    n_iter = 24  # change to larger for more thorough tuning if compute/time available

    for name, cfg in candidates.items():
        print(f"\nTuning {name} ...")
        model = cfg["model"]
        param_dist = cfg["param_dist"]

        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        start = time.time()
        # use scaled features for consistent comparison
        rs.fit(X_train_scaled, y_train)
        elapsed = time.time() - start
        print(f"{name} tuned (time {elapsed:.1f}s) - best CV MAE: {-rs.best_score_:.4f}")

        best_model = rs.best_estimator_
        y_pred_test = best_model.predict(X_test_scaled)
        metrics = evaluate_metrics(y_test.values, y_pred_test)
        print(f"{name} test MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape_percent']:.2f}%")

        # save candidate best
        joblib.dump(best_model, os.path.join(model_dir, f"{name}_best.joblib"))
        results[name] = {
            "search": rs,
            "best_estimator": best_model,
            "cv_mae": -rs.best_score_,
            "test_metrics": metrics
        }

    # choose best by test MAE
    best_name = min(results.keys(), key=lambda k: results[k]["test_metrics"]["mae"])
    best_model = results[best_name]["best_estimator"]
    best_metrics = results[best_name]["test_metrics"]

    # save final model
    final_model_path = os.path.join(model_dir, "best_electricity_model.joblib")
    joblib.dump(best_model, final_model_path)

    # plot actual vs predicted
    y_best_pred = best_model.predict(X_test_scaled)
    plot_actual_vs_pred(test_df, y_test.values, y_best_pred, outpath=os.path.join(model_dir, "actual_vs_pred.png"))

    # feature importance
    try:
        fi = None
        if hasattr(best_model, "feature_importances_"):
            fi = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            fi = np.abs(best_model.coef_)

        if fi is not None:
            fi_series = pd.Series(fi, index=feature_cols).sort_values(ascending=False)
            fi_series.head(40).to_csv(os.path.join(model_dir, "feature_importance.csv"), header=["importance"])
            plt.figure(figsize=(8, 6))
            fi_series.head(30).plot(kind="barh")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "feature_importance.png"))
            plt.close()
    except Exception:
        pass

    # summaryyy
    print("\n=== model comparison summary ===")
    for k, v in results.items():
        tm = v["test_metrics"]
        print(f"{k}: CV_MAE={v['cv_mae']:.4f} | Test_MAE={tm['mae']:.4f} | MAPE={tm['mape_percent']:.2f}%")

    print(f"\nSelected best model: {best_name} (saved at {final_model_path})")
    return {
        "best_name": best_name,
        "best_model_path": final_model_path,
        "results": results,
        "feature_cols": feature_cols,
        "scaler_path": os.path.join(model_dir, "scaler.joblib"),
    }


# ----------------------
# helper
# ----------------------
def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape_percent": float(mape)}


# plot ->>>>actual vs predicted
def plot_actual_vs_pred(test_df, y_true, y_pred, outpath: str):
    plt.figure(figsize=(12, 4))
    plt.plot(test_df["timestamp"], y_true, label="Actual")
    plt.plot(test_df["timestamp"], y_pred, label="Predicted")
    plt.xlabel("Time"); plt.ylabel("Demand"); plt.title("Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ----------------------
# # prediction helper
# ----------------------
def predict_for_timestamp(timestamp_str: str) -> float:
    """
    Predict demand for a timestamp.
    - If timestamp exists in dataset after feature creation: return model prediction for that timestamp.
    - If timestamp is in the future: build features from recent history and predict.
    """
    df = load_dataset(DATA_FILE)
    df_feat = create_advanced_features(df)
    # check exact timestamp case
    ts = pd.to_datetime(timestamp_str)
    row = df_feat[df_feat["timestamp"] == ts]
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    model_path = os.path.join(MODEL_DIR, "best_electricity_model.joblib")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Trained model or scaler not found. Run training first.")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    if not row.empty:
        feature_cols = [c for c in df_feat.columns if c not in ("timestamp", "demand")]
        X_row = row[feature_cols]
        Xs = scaler.transform(X_row)
        return float(model.predict(Xs)[0])

    # future timestamp: need to construct feature vector from latest history
    last = df.copy().set_index("timestamp").sort_index()
    needed_history = 168  # 1 week
    if len(last) < needed_history:
        raise ValueError("Not enough history to build lag/rolling features for forecast.")

    # compute lags and rolling stats based on last history
    last_sorted = last
    lag_values = {}
    for lag in [1, 24, 48, 72, 168]:
        lag_values[f"lag_{lag}"] = last_sorted["demand"].iloc[-lag]

    roll_values = {}
    for w in [24, 48, 72, 168]:
        roll_values[f"roll_mean_{w}"] = last_sorted["demand"].iloc[-w:].mean()
        roll_values[f"roll_std_{w}"] = last_sorted["demand"].iloc[-w:].std()

    feature_row = {
        "hour": ts.hour,
        "dayofweek": ts.dayofweek,
        "day": ts.day,
        "month": ts.month,
        "is_weekend": int(ts.dayofweek >= 5),
        **lag_values,
        **{k: roll_values[k] for k in roll_values},
        "time_idx": last_sorted.shape[0],
    }

    # align with training columns
    feature_cols = [c for c in df_feat.columns if c not in ("timestamp", "demand")]
    X_row_df = pd.DataFrame([feature_row])[feature_cols]
    Xs = scaler.transform(X_row_df)
    pred = float(model.predict(Xs)[0])
    return pred


# ----------------------
# lets run script
# ----------------------
if __name__ == "__main__":
    start_time = time.time()
    data = load_dataset(DATA_FILE)
    print(f"Data loaded: {len(data)} rows. Range: {data['timestamp'].min()} to {data['timestamp'].max()}")

    train_info = train_and_tune(data)
    elapsed = time.time() - start_time
    print(f"\nFinished. Elapsed time: {elapsed:.0f}s")

    
    # print("Example: predict for existing timestamp in dataset:")
    # print(predict_for_timestamp("2025-11-01 12:00:00"))
    # print("Example: predict future timestamp:")
    # print(predict_for_timestamp("2026-01-01 12:00:00"))
