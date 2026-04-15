import io
import json
import logging
import os
import traceback

import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------------- ENV ----------------
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_llm_master.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds_llm")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


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


def clean_numeric(series):
    s = series.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def compute_metrics(y_true, y_pred):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    bias = float((y_pred - y_true).mean())

    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = float((np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]).mean()) * 100.0)
    else:
        mape = None

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape,
    }


def run_once(dry_run=False):
    client = storage.Client(project=PROJECT_ID)
    df = read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage", "color"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ---- prepare date split ----
    df["scraped_at_dt_utc"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)

    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]

    df["date_local"] = df["scraped_at_local"].dt.date

    # ---- clean numerics ----
    df["price_num"] = clean_numeric(df["price"])
    df["year_num"] = clean_numeric(df["year"])
    df["mileage_num"] = clean_numeric(df["mileage"])

    df = df.dropna(subset=["price_num", "year_num", "mileage_num"]).copy()

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {
            "status": "noop",
            "reason": "need at least two distinct dates",
            "dates": [str(d) for d in unique_dates],
        }

    today_local = unique_dates[-1]
    train_df = df[df["date_local"] < today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    if len(train_df) < 40:
        return {
            "status": "noop",
            "reason": "too few training rows",
            "train_rows": int(len(train_df)),
        }

    # ---- features ----
    target = "price_num"
    cat_cols = ["make", "model", "color"]
    num_cols = ["year_num", "mileage_num"]
    feats = cat_cols + num_cols

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    base_pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("model", DecisionTreeRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "model__max_depth": [4, 6, 8, 10, 12],
        "model__min_samples_leaf": [5, 10, 20],
    }

    grid = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
    )

    X_train = train_df[feats]
    y_train = train_df[target]
    grid.fit(X_train, y_train)

    pipe = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_mae = float(-grid.best_score_)

    logging.info("Best params: %s", json.dumps(best_params))
    logging.info("Best CV MAE: %.4f", best_cv_mae)

    # ---- predictions on today's holdout ----
    preds_df = pd.DataFrame()
    holdout_metrics = None

    if not holdout_df.empty:
        X_holdout = holdout_df[feats]
        y_holdout = holdout_df[target]
        y_pred = pipe.predict(X_holdout)

        keep_cols = ["post_id", "scraped_at", "make", "model", "year", "mileage", "color", "price"]
        preds_df = holdout_df[keep_cols].copy()
        preds_df["actual_price"] = y_holdout.values
        preds_df["pred_price"] = np.round(y_pred, 2)

        holdout_metrics = compute_metrics(y_holdout, y_pred)

    # ---- permutation importance ----
    # Use holdout if available, else train
    if not holdout_df.empty:
        X_imp = holdout_df[feats]
        y_imp = holdout_df[target]
    else:
        X_imp = train_df[feats]
        y_imp = train_df[target]

    perm = permutation_importance(
        pipe,
        X_imp,
        y_imp,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    importance_df = pd.DataFrame(
        {
            "feature": feats,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # ---- write outputs ----
    now_utc = pd.Timestamp.now(tz="UTC")
    out_prefix = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}"

    preds_key = f"{out_prefix}/preds.csv"
    importance_key = f"{out_prefix}/importance.csv"
    metrics_key = f"{out_prefix}/metrics.json"

    metrics_payload = {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "best_cv_mae": best_cv_mae,
        "best_params": best_params,
        "features_used": feats,
        "preds_key": preds_key if len(preds_df) > 0 else None,
        "importance_key": importance_key,
        "holdout_metrics": holdout_metrics,
        "data_key": DATA_KEY,
        "timezone": TIMEZONE,
    }

    if not dry_run:
        if len(preds_df) > 0:
            write_csv_to_gcs(client, GCS_BUCKET, preds_key, preds_df)
            logging.info("Wrote predictions to gs://%s/%s", GCS_BUCKET, preds_key)

        write_csv_to_gcs(client, GCS_BUCKET, importance_key, importance_df)
        logging.info("Wrote permutation importance to gs://%s/%s", GCS_BUCKET, importance_key)

        write_json_to_gcs(client, GCS_BUCKET, metrics_key, metrics_payload)
        logging.info("Wrote metrics to gs://%s/%s", GCS_BUCKET, metrics_key)
    else:
        logging.info("Dry run enabled; skipping output writes")

    return metrics_payload


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
