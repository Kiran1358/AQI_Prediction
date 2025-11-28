# backend/train_regressor_safe.py
"""
Robust training script for CPCB city_day.csv.
- Detects columns automatically
- Converts to numeric (coerce)
- Fills/handles missing values safely
- Trains RandomForest and saves artifacts
"""

import os, argparse, json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

def detect_columns(df):
    # normalize names
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # detect AQI column (any column name that contains 'aqi' case-insensitive)
    aqi_col = None
    for c in df.columns:
        if 'aqi' in str(c).lower():
            aqi_col = c
            break

    # detect city column
    city_col = None
    for c in ['City','city','CITY','Station','station']:
        if c in df.columns:
            city_col = c
            break

    # detect date column
    date_col = None
    for c in ['Date','date','DATE','timestamp','Timestamp','time','Time']:
        if c in df.columns:
            date_col = c
            break

    # pollutant candidates (common CPCB names). Keep those present.
    pollutant_candidates = ["PM2.5","PM10","NO2","SO2","CO","O3","NH3"]
    pollutants = [c for c in pollutant_candidates if c in df.columns]

    return city_col, date_col, aqi_col, pollutants

def clean_dataframe(df):
    city_col, date_col, aqi_col, pollutants = detect_columns(df)
    print("Detected columns -> city:", city_col, ", date:", date_col, ", aqi:", aqi_col)
    print("Detected pollutant columns:", pollutants)

    if aqi_col is None:
        raise RuntimeError("AQI column not found. Please check city_day.csv headers and re-run.")

    # rename standardized columns
    if city_col:
        df = df.rename(columns={city_col: "City"})
    else:
        df["City"] = "Unknown"

    if date_col:
        try:
            df["Date"] = pd.to_datetime(df[date_col], errors='coerce')
        except Exception:
            df["Date"] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df["Date"] = pd.NaT

    df = df.rename(columns={aqi_col: "AQI"})

    # Drop rows without AQI
    df = df.dropna(subset=["AQI"]).reset_index(drop=True)
    print("Rows after dropping missing AQI:", len(df))

    # If no pollutant columns detected, try to discover numeric columns that look like pollutants
    if not pollutants:
        # choose numeric columns except Date, AQI
        possible = []
        for c in df.columns:
            if c in ["City","Date","AQI"]:
                continue
            if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object:
                possible.append(c)
        # Heuristic: pick up to 7 numeric-looking columns excluding obvious metadata
        pollutants = possible[:7]
        print("Fallback pollutant selection:", pollutants)

    # Convert pollutant columns to numeric
    for p in pollutants:
        df[p] = pd.to_numeric(df[p], errors='coerce')

    # Fill per-city using group transform (keeps index alignment). Use ffill->bfill.
    for p in pollutants:
        try:
            df[p] = df.groupby('City')[p].transform(lambda s: s.ffill().bfill())
        except Exception as e:
            print(f"Warning: group transform failed for {p}: {e}. Using global fill.")
            df[p] = df[p].ffill().bfill()

    # Drop rows that still have missing pollutant values
    df = df.dropna(subset=pollutants).reset_index(drop=True)
    print("Rows after filling & dropping pollutant NaNs:", len(df))

    # Ensure numeric AQI
    df["AQI"] = pd.to_numeric(df["AQI"], errors='coerce')
    df = df.dropna(subset=["AQI"]).reset_index(drop=True)
    print("Rows after ensuring numeric AQI:", len(df))

    return df, pollutants

def train(input_path, model_dir):
    print("Loading:", input_path)
    df = pd.read_csv(input_path)
    print("Raw rows:", len(df))

    df_clean, pollutants = clean_dataframe(df)
    if len(df_clean) < 100:
        print("Warning: cleaned dataframe has very few rows:", len(df_clean))

    X = df_clean[pollutants].values.astype(np.float64)
    y = df_clean["AQI"].values.astype(np.float64)

    # final sanity: remove rows with nan/inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    print("Final training rows:", X.shape[0], "features:", X.shape[1])

    if X.shape[0] < 50:
        raise RuntimeError("Not enough samples for training after cleaning. Need >=50 rows.")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # train RandomForest
    reg = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    reg.fit(X_train_s, y_train)

    # predict & evaluate (use numpy for RMSE to avoid sklearn wrapper edge-cases)
    y_pred = reg.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    # --- ADDED: compute R^2 as an "accuracy" metric for regression and print training R^2 too
    try:
        r2_test = r2_score(y_test, y_pred)
    except Exception:
        r2_test = None

    try:
        r2_train = reg.score(X_train_s, y_train)
    except Exception:
        r2_train = None

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    if r2_test is not None:
        print(f"R^2 (test) : {r2_test:.4f}  (interpreted as regression 'accuracy')")
    if r2_train is not None:
        print(f"R^2 (train): {r2_train:.4f}")

    # cross-val MAE (5-fold)
    try:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = -1 * np.mean(cross_val_score(reg, scaler.transform(X), y, cv=cv,
                                                  scoring='neg_mean_absolute_error', n_jobs=-1))
    except Exception as e:
        print("Cross-val error:", e)
        cv_scores = None

    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(reg, os.path.join(model_dir, "aqi_regressor.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
        json.dump(pollutants, f)
    metrics = {"regression": {"MAE": float(mae), "RMSE": float(rmse)}}
    if cv_scores is not None:
        metrics["regression"]["CV_MAE"] = float(cv_scores)
    # include R^2 in metrics.json as well
    if r2_test is not None:
        metrics["regression"]["R2_test"] = float(r2_test)
    if r2_train is not None:
        metrics["regression"]["R2_train"] = float(r2_train)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model artifacts to", model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--models", "-m", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    args = parser.parse_args()
    train(args.input, args.models)
