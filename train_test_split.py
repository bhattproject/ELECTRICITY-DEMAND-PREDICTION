
"""
Train/Test Split for Electricity Demand Prediction

This script performs a comprehensive train/validation/test split
for time series data, with multiple options for:
- Single split or multiple rolling splits
- Scaling features
- Feature/target separation
- Saving datasets for reproducibility
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import os


DATA_PATH = "Electricity_Dataset_Features.csv"
df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)


TARGET_COL = 'Demand'      # Column to predict
TEST_SIZE = 0.2            # Fraction of data for test
VAL_SIZE = 0.1             # Fraction of train for validation
SCALER_TYPE = 'standard'   # 'standard' or 'minmax'
OUTPUT_DIR = "data_splits"
ROLLING_SPLITS = True       # True for multiple rolling splits

os.makedirs(OUTPUT_DIR, exist_ok=True)
FEATURE_COLS = [c for c in df.columns if c not in ['Datetime', TARGET_COL]]

X = df[FEATURE_COLS]
y = df[TARGET_COL]

if SCALER_TYPE == 'standard':
    scaler = StandardScaler()
elif SCALER_TYPE == 'minmax':
    scaler = MinMaxScaler()
else:
    raise ValueError("SCALER_TYPE must be 'standard' or 'minmax'")

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)


def time_series_split(X, y, test_size=0.2, val_size=0.1):
    """
    Perform a train/validation/test split with time series data.
    Returns X_train, X_val, X_test, y_train, y_val, y_test
    """
    n_total = len(X)
    n_test = int(n_total * test_size)
    n_train_val = n_total - n_test
    n_val = int(n_train_val * val_size)
    n_train = n_train_val - n_val

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val = X.iloc[n_train:n_train+n_val]
    y_val = y.iloc[n_train:n_train+n_val]
    X_test = X.iloc[n_train+n_val:]
    y_test = y.iloc[n_train+n_val:]

    return X_train, X_val, X_test, y_train, y_val, y_test

# Single split
X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(X_scaled, y, TEST_SIZE, VAL_SIZE)

# Save splits
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_val.to_csv(os.path.join(OUTPUT_DIR, "X_val.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_val.to_csv(os.path.join(OUTPUT_DIR, "y_val.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
print(f"Train/Val/Test split saved in {OUTPUT_DIR}")


if ROLLING_SPLITS:
    print("\nCreating rolling splits for time series cross-validation...")

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rolling_dir = os.path.join(OUTPUT_DIR, "rolling_splits")
    os.makedirs(rolling_dir, exist_ok=True)

    split_num = 1
    for train_index, test_index in tscv.split(X_scaled):
        X_train_r, X_test_r = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_train_r, y_test_r = y.iloc[train_index], y.iloc[test_index]

       
        n_val_r = int(len(X_train_r) * VAL_SIZE)
        X_train_final = X_train_r.iloc[:-n_val_r]
        X_val_final = X_train_r.iloc[-n_val_r:]
        y_train_final = y_train_r.iloc[:-n_val_r]
        y_val_final = y_train_r.iloc[-n_val_r:]

        X_train_final.to_csv(os.path.join(rolling_dir, f"X_train_split{split_num}.csv"), index=False)
        X_val_final.to_csv(os.path.join(rolling_dir, f"X_val_split{split_num}.csv"), index=False)
        X_test_r.to_csv(os.path.join(rolling_dir, f"X_test_split{split_num}.csv"), index=False)
        y_train_final.to_csv(os.path.join(rolling_dir, f"y_train_split{split_num}.csv"), index=False)
        y_val_final.to_csv(os.path.join(rolling_dir, f"y_val_split{split_num}.csv"), index=False)
        y_test_r.to_csv(os.path.join(rolling_dir, f"y_test_split{split_num}.csv"), index=False)

        print(f"Rolling split {split_num} saved.")
        split_num += 1
print("\nSplit summary:")
pr
