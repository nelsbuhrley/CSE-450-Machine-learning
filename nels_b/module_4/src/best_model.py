"""★ BEST current bike-demand predictor ★

Equal-weight blend of the V2 Keras MLP and XGBoost. On the mini holdout:
RMSE 127.76, R² 0.877 — beats every single model and every learned stacker we tried.

    final = ( V2_MLP_total + XGBoost_total ) / 2

The blend wins because XGBoost is strong exactly where the MLP is weak (rush-hour
peaks) and its errors decorrelate (0.74); equal weighting is the robust combiner
(learned weights overfit the train/holdout distribution shift — see meta_blend.py).

Self-contained artifacts live in best_model/ (MLP, two XGB models, scalers, feature
order), so prediction needs no retraining.

Usage:
  python best_model.py --fit                       # train + save the XGB half (MLP already trained)
  python best_model.py --predict FILE.csv --out OUT.csv
  python best_model.py                             # fit if needed, then score the mini holdout
"""

import argparse
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import module4 as m
import paths

DIR = paths.BEST_MODEL
MLP_SRC = os.path.join(paths.MODELS_V2, "model1_b256.keras")
MLP = f"{DIR}/mlp.keras"
XGB_C = f"{DIR}/xgb_casual.json"
XGB_R = f"{DIR}/xgb_registered.json"
XSCALER = f"{DIR}/x_scaler.joblib"
YSCALER = f"{DIR}/y_scaler.joblib"
FEAT = f"{DIR}/feature_cols.json"
HOLDOUT = paths.MINI_HOLDOUT
ANSWERS_URL = paths.ANSWERS_URL

XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.02, max_depth=8, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
    objective="reg:squarederror", eval_metric="rmse", early_stopping_rounds=50,
    n_jobs=8, random_state=m.RANDOM_STATE,
)


def _feature_cols(engineered):
    return [c for c in engineered.columns if c not in ("dteday", "hr", "dow", "casual", "registered")]


def fit_and_save():
    """Train the XGB half on the V2 pipeline, save all artifacts. MLP is reused as-is."""
    os.makedirs(DIR, exist_ok=True)
    data = m.load_and_engineer_features(m.DATA_URL)
    feat_cols = _feature_cols(data)
    X, y = data[feat_cols], data[["casual", "registered"]]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=m.RANDOM_STATE)

    x_scaler = MinMaxScaler().fit(X_tr)
    y_scaler = MinMaxScaler().fit(np.log1p(y_tr))  # MLP target space (log1p)

    v = int(len(X_tr) * 0.15)
    for col, path in (("casual", XGB_C), ("registered", XGB_R)):
        reg = xgb.XGBRegressor(**XGB_PARAMS)
        reg.fit(X_tr.iloc[v:], y_tr[col].iloc[v:],
                eval_set=[(X_tr.iloc[:v], y_tr[col].iloc[:v])], verbose=False)
        reg.save_model(path)

    joblib.dump(x_scaler, XSCALER)
    joblib.dump(y_scaler, YSCALER)
    with open(FEAT, "w") as f:
        json.dump(feat_cols, f)
    shutil.copyfile(MLP_SRC, MLP)
    print(f"★ saved best predictor to {DIR}/ (mlp.keras, xgb_casual/registered.json, scalers, feature_cols)")


def predict(input_csv: str) -> np.ndarray:
    """Equal-weight blend of V2 MLP and XGBoost totals for rows in input_csv."""
    with open(FEAT) as f:
        feat_cols = json.load(f)
    x_scaler = joblib.load(XSCALER)
    y_scaler = joblib.load(YSCALER)
    mlp = tf.keras.models.load_model(MLP, compile=False)
    rc, rr = xgb.XGBRegressor(), xgb.XGBRegressor()
    rc.load_model(XGB_C)
    rr.load_model(XGB_R)

    X = m.load_and_engineer_features(input_csv)[feat_cols]
    mlp_total = m.invert_targets(y_scaler, mlp.predict(x_scaler.transform(X), verbose=0)).sum(axis=1)
    xgb_total = np.clip(np.round(rc.predict(X) + rr.predict(X), 1), 0, None)
    return np.clip(np.round((mlp_total + xgb_total) / 2.0, 1), 0, None)


def _artifacts_exist():
    return all(os.path.exists(p) for p in (MLP, XGB_C, XGB_R, XSCALER, YSCALER, FEAT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="train + save the predictor")
    ap.add_argument("--predict", metavar="CSV", help="input CSV to score")
    ap.add_argument("--out", default=paths.expl("best_predictions.csv"), help="output predictions CSV")
    args = ap.parse_args()
    m.configure_cpu_parallelism()

    if args.fit or not _artifacts_exist():
        fit_and_save()

    target = args.predict or HOLDOUT
    preds = predict(target)
    pd.DataFrame({"predictions": preds}).to_csv(args.out, index=False)
    print(f"wrote {len(preds)} predictions -> {args.out}")

    if args.predict is None:  # scored the mini holdout — report metrics
        ans = pd.read_csv(ANSWERS_URL)
        actual = (ans["casual"] + ans["registered"]).to_numpy()
        print(f"\n★ BEST predictor on mini holdout:  RMSE {root_mean_squared_error(actual, preds):.2f}"
              f"   R² {r2_score(actual, preds):.3f}")


if __name__ == "__main__":
    main()
