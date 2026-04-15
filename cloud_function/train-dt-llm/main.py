import os
import io
import json
import logging

import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# ===== ENV =====
PROJECT_ID = os.getenv("PROJECT_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")

DATA_KEY = "structured/datasets/listings_llm_master.csv"
OUTPUT_PREFIX = "preds_llm"

logging.basicConfig(level=logging.INFO)


# ===== GCS HELPERS =====
def read_csv_from_gcs(client, bucket, key):
    b = client.bucket(bucket)
    blob = b.blob(key)

    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")

    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def write_csv_to_gcs(client, bucket, key, df):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


def write_json_to_gcs(client, bucket, key, payload):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(
        json.dumps(payload, indent=2),
        content_type="application/json"
    )


# ===== CLEANING =====
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce"
    )


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = None

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape
    }


# ===== MAIN PIPELINE =====
def run_pipeline():
    client = storage.Client(project=PROJECT_ID)

    logging.info("Reading dataset...")
    df = read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    # ===== CLEAN DATA =====
    df["price"] = clean_numeric(df["price"])
    df["year"] = clean_numeric(df["year"])
    df["mileage"] = clean_numeric(df["mileage"])

    feature_cols = ["year", "mileage", "make", "model", "color"]
    needed = ["price"] + feature_cols
    df = df.dropna(subset=["price", "year", "mileage"]).copy()

    # keep text cols usable
    for col in ["make", "model", "color"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
        else:
            df[col] = "unknown"

    # ===== FEATURES =====
    X_raw = df[feature_cols].copy()
    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===== MODEL WITH TUNING =====
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
    }

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    # predictions for full dataset
    preds = model.predict(X)
    df["pred_price"] = np.round(preds, 2)

    # holdout metrics
    test_preds = model.predict(X_test)
    metrics = compute_metrics(y_test, test_preds)
    metrics["best_params"] = grid.best_params_
    metrics["best_cv_mae"] = float(-grid.best_score_)
    metrics["rows_total"] = int(len(df))
    metrics["train_rows"] = int(len(X_train))
    metrics["test_rows"] = int(len(X_test))
    metrics["features_used"] = feature_cols
    metrics["expanded_feature_count"] = int(X.shape[1])

    # permutation importance on holdout
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error"
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    # ===== WRITE OUTPUT =====
    run_id = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    base_prefix = f"{OUTPUT_PREFIX}/{run_id}"

    preds_key = f"{base_prefix}/preds.csv"
    importance_key = f"{base_prefix}/importance.csv"
    metrics_key = f"{base_prefix}/metrics.json"

    logging.info(f"Writing predictions to {preds_key}")
    write_csv_to_gcs(client, GCS_BUCKET, preds_key, df)

    logging.info(f"Writing importance to {importance_key}")
    write_csv_to_gcs(client, GCS_BUCKET, importance_key, importance_df)

    logging.info(f"Writing metrics to {metrics_key}")
    write_json_to_gcs(client, GCS_BUCKET, metrics_key, metrics)

    return {
        "status": "success",
        "preds_output": preds_key,
        "importance_output": importance_key,
        "metrics_output": metrics_key,
        "best_params": grid.best_params_,
        "best_cv_mae": float(-grid.best_score_)
    }


# ===== CLOUD FUNCTION ENTRY =====
def train_dt_http(request):
    try:
        output = run_pipeline()
        return (
            json.dumps(output),
            200,
            {"Content-Type": "application/json"},
        )

    except Exception as e:
        logging.exception("Error in pipeline")
        return (
            json.dumps({
                "status": "error",
                "error": str(e)
            }),
            500,
            {"Content-Type": "application/json"},
        )
