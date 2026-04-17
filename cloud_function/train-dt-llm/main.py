import io
import json
import logging
import os

import functions_framework
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# ===== ENV =====
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_llm_master.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds_llm")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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
        content_type="application/json",
    )


def write_png_to_gcs(client, bucket, key, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_file(buf, content_type="image/png")
    buf.close()


# ===== CLEANING =====
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0
        )
    else:
        mape = None

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape,
    }


# ===== MAIN PIPELINE =====
def run_pipeline():
    client = storage.Client(project=PROJECT_ID)

    logging.info("Reading dataset from GCS...")
    df = read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    # ===== CLEAN DATA =====
    df["price"] = clean_numeric(df["price"])
    df["year"] = clean_numeric(df["year"])
    df["mileage"] = clean_numeric(df["mileage"])

    df = df.dropna(subset=["price", "year", "mileage"]).copy()

    # Fill text features safely
    for col in ["make", "model", "color"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
        else:
            df[col] = "unknown"

    # ===== FEATURES =====
    feature_cols = ["year", "mileage", "make", "model", "color"]
    X_raw = df[feature_cols].copy()
    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["price"]

    if len(df) < 20:
        raise ValueError("Not enough rows to train a model")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ===== MODEL WITH TUNING =====
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
    }

    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    logging.info("Running GridSearchCV...")
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    logging.info("Best params: %s", json.dumps(grid.best_params_))
    logging.info("Best CV MAE: %.4f", float(-grid.best_score_))

    # ===== PREDICTIONS =====
    preds_all = model.predict(X)
    df["pred_price"] = np.round(preds_all, 2)

    test_preds = model.predict(X_test)
    metrics = compute_metrics(y_test, test_preds)
    metrics["best_params"] = grid.best_params_
    metrics["best_cv_mae"] = float(-grid.best_score_)
    metrics["rows_total"] = int(len(df))
    metrics["train_rows"] = int(len(X_train))
    metrics["test_rows"] = int(len(X_test))
    metrics["features_used_raw"] = feature_cols
    metrics["expanded_feature_count"] = int(X.shape[1])

    # ===== PERMUTATION IMPORTANCE =====
    logging.info("Computing permutation importance...")
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # ===== PDP (TOP 3 FEATURES) =====
    logging.info("Generating PDP plot...")
    top_features = importance_df["feature"].head(3).tolist()

    fig, axes = plt.subplots(1, len(top_features), figsize=(6 * len(top_features), 4))
    if len(top_features) == 1:
        axes = [axes]

    PartialDependenceDisplay.from_estimator(
        model,
        X_test,
        features=top_features,
        ax=axes,
    )

    # ===== WRITE OUTPUTS =====
    run_id = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    base_prefix = f"{OUTPUT_PREFIX}/{run_id}"

    preds_key = f"{base_prefix}/preds.csv"
    importance_key = f"{base_prefix}/importance.csv"
    metrics_key = f"{base_prefix}/metrics.json"
    pdp_key = f"{base_prefix}/pdp.png"

    logging.info("Writing predictions to %s", preds_key)
    write_csv_to_gcs(client, GCS_BUCKET, preds_key, df)

    logging.info("Writing permutation importance to %s", importance_key)
    write_csv_to_gcs(client, GCS_BUCKET, importance_key, importance_df)

    logging.info("Writing metrics to %s", metrics_key)
    write_json_to_gcs(client, GCS_BUCKET, metrics_key, metrics)

    logging.info("Writing PDP plot to %s", pdp_key)
    write_png_to_gcs(client, GCS_BUCKET, pdp_key, fig)
    plt.close(fig)

    return {
        "status": "success",
        "preds_output": preds_key,
        "importance_output": importance_key,
        "metrics_output": metrics_key,
        "pdp_output": pdp_key,
        "best_params": grid.best_params_,
        "best_cv_mae": float(-grid.best_score_),
    }


# ===== CLOUD FUNCTION ENTRY =====
@functions_framework.http
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
            json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                }
            ),
            500,
            {"Content-Type": "application/json"},
        )
