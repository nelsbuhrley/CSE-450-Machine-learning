"""Original V1 bike pipeline, kept separate so V1 models can be scored after
module4.py moved to the V2 pipeline (day-of-week feats, log1p targets, single scale).

V1 = 19 features, num_cols min-max scaled inside feature engineering, raw-count
targets. A holdout is scaled with the training-fit num_scaler for consistency.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import paths
from module4 import DATA_URL, RANDOM_STATE, REF_DATE

NUM_COLS = ["temp_c", "feels_like_c", "hum", "windspeed", "total_hours", "weathersit"]
DROP_COLS = ["dteday", "hr", "casual", "registered"]


def engineer_v1(url: str, num_scaler: MinMaxScaler | None = None):
    df = pd.read_csv(url)

    df["hour_sin"] = np.sin(df["hr"] * (2.0 * np.pi / 24))
    df["hour_cos"] = np.cos(df["hr"] * (2.0 * np.pi / 24))
    df["season_sin"] = np.sin(df["season"] * (2.0 * np.pi / 4))
    df["season_cos"] = np.cos(df["season"] * (2.0 * np.pi / 4))

    df["dteday"] = pd.to_datetime(df["dteday"])
    df["day"] = df["dteday"].dt.day
    df["month"] = df["dteday"].dt.month
    df["year"] = df["dteday"].dt.year

    df["total_hours"] = (df["dteday"] - REF_DATE).dt.total_seconds() / 3600 + df["hr"]
    df["time_sin"] = np.sin(df["total_hours"] * (2.0 * np.pi / 24))
    df["time_cos"] = np.cos(df["total_hours"] * (2.0 * np.pi / 24))

    scaler = num_scaler if num_scaler is not None else MinMaxScaler().fit(df[NUM_COLS])
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    p = 2
    df["weather_intensity"] = (
        (df["temp_c"] ** p + df["hum"] ** p + df["windspeed"] ** p + df["weathersit"] ** p) / 3
    ) ** (1 / p)

    return df, scaler


def prepare_v1():
    """Reconstruct V1 scalers + test split (deterministic, RANDOM_STATE=42)."""
    data, num_scaler = engineer_v1(DATA_URL)
    y = data[["casual", "registered"]]
    x = data.drop(columns=DROP_COLS)

    X_tr, X_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE)
    x_scaler = MinMaxScaler().fit(X_tr)
    y_scaler = MinMaxScaler().fit(y_tr)  # raw counts (V1 trained without log1p)
    return x_scaler, y_scaler, x_scaler.transform(X_te), y_te, num_scaler


def holdout_features_v1(x_scaler, num_scaler):
    df, _ = engineer_v1(paths.MINI_HOLDOUT, num_scaler=num_scaler)
    x = df.drop(columns=["dteday", "hr"])
    return x_scaler.transform(x)


def invert_v1(y_scaler, pred_scaled):
    """V1 targets are raw counts (no log): just inverse-scale, clip, round."""
    return np.clip(np.round(y_scaler.inverse_transform(pred_scaled), 1), 0, None)
