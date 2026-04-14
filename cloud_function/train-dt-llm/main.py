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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_llm_master.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds_llm")

logging.basicConfig(level=logging.INFO)


def read_csv(client, bucket, key):
    b = client.bucket(bucket)
    blob = b.blob(key)
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def write_csv(client, bucket, key, df):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


def run_once():
    client = storage.Client(project=PROJECT_ID)
    df = read_csv(client, GCS_BUCKET, DATA_KEY)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

    df = df.dropna(subset=["price"])

    X = df[["make", "model", "color", "year", "mileage"]]
    y = df["price"]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), ["year", "mileage"]),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), ["make", "model", "color"])
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("model", DecisionTreeRegressor())
    ])

    grid = GridSearchCV(
        pipe,
        {
            "model__max_depth": [4, 6, 8],
            "model__min_samples_leaf": [5, 10]
        },
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    grid.fit(X, y)

    preds = grid.predict(X)

    out = df.copy()
    out["pred_price"] = preds

    key = f"{OUTPUT_PREFIX}/test/preds.csv"
    write_csv(client, GCS_BUCKET, key, out)

    return {
        "status": "ok",
        "best_params": grid.best_params_
    }


def train_dt_http(request):
    try:
        result = run_once()
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error(str(e))
        logging.error(traceback.format_exc())
        return (json.dumps({"error": str(e)}), 500)
