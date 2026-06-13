"""★ BEST v3 bike-demand predictor — holiday-aware ★

Same equal-weight MLP+XGBoost blend as best_model.py, but both base learners are
retrained on the V3 feature set (holiday_features.py), which adds holiday-PROXIMITY
features: day-before / day-after / bridge-day flags, distance-to-nearest-day-off, and
an hour×holiday-week interaction so the model can dampen the commute peak on working
days that sit inside a holiday neighbourhood (the Thanksgiving-week over-prediction).

Trains into fresh folders (models_v3/, best_model_v3/) — the V2 artifacts in models_v2/
and best_model/ are left untouched.

Usage:
  python best_model_v3.py --fit                     # train MLP + XGB on V3 features, save
  python best_model_v3.py --predict FILE.csv --out OUT.csv
  python best_model_v3.py                           # fit if needed, then score the mini holdout
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
from holiday_features import engineer_v3, feature_cols_v3

DIR = paths.BEST_MODEL_V3
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


def fit_and_save():
    """Train MLP + XGB on the V3 (holiday-aware) features and save all artifacts."""
    os.makedirs(DIR, exist_ok=True)
    data = engineer_v3(m.DATA_URL)
    feat_cols = feature_cols_v3(data)
    X, y = data[feat_cols], data[["casual", "registered"]]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=m.RANDOM_STATE)

    x_scaler = MinMaxScaler().fit(X_tr)
    y_scaler = MinMaxScaler().fit(np.log1p(y_tr))
    Xtr_s, Xte_s = x_scaler.transform(X_tr), x_scaler.transform(X_te)
    ytr_s = y_scaler.transform(np.log1p(y_tr))

    # Same architecture as the V2 best (effort 6, batch 256) so the only change is the features.
    config = m.make_configs(6, [{"batch_size": 256, "dropout": 0.2}], base_lr=0.05, ref_batch=1024)[0]
    res = m.train_one(
        Xtr_s, ytr_s, Xte_s, y_te, y_scaler, config,
        position=0, desc="V3 MLP (holiday-aware)", save_dir=paths.MODELS_V3, name="model1_b256_v3",
    )
    print(f"V3 MLP   X_test RMSE {res['rmse']:.2f}  R² {res['r2']:.3f}  ({res['epochs_trained']} epochs)")
    shutil.copyfile(res["model_path"], MLP)

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
    print(f"★ saved V3 predictor to {DIR}/ ({len(feat_cols)} features)")


def _load():
    with open(FEAT) as f:
        feat_cols = json.load(f)
    x_scaler = joblib.load(XSCALER)
    y_scaler = joblib.load(YSCALER)
    mlp = tf.keras.models.load_model(MLP, compile=False)
    rc, rr = xgb.XGBRegressor(), xgb.XGBRegressor()
    rc.load_model(XGB_C)
    rr.load_model(XGB_R)
    return feat_cols, x_scaler, y_scaler, mlp, rc, rr


def predict_parts(input_csv: str):
    """Return (mlp_total, xgb_total, blend) for rows in input_csv."""
    feat_cols, x_scaler, y_scaler, mlp, rc, rr = _load()
    X = engineer_v3(input_csv)[feat_cols]
    mlp_total = m.invert_targets(y_scaler, mlp.predict(x_scaler.transform(X), verbose=0)).sum(axis=1)
    xgb_total = np.clip(np.round(rc.predict(X) + rr.predict(X), 1), 0, None)
    blend = np.clip(np.round((mlp_total + xgb_total) / 2.0, 1), 0, None)
    return mlp_total, xgb_total, blend


def predict(input_csv: str) -> np.ndarray:
    return predict_parts(input_csv)[2]


def _artifacts_exist():
    return all(os.path.exists(p) for p in (MLP, XGB_C, XGB_R, XSCALER, YSCALER, FEAT))


def _score(name, pred, actual):
    return f"{name:28s} RMSE {root_mean_squared_error(actual, pred):7.2f}   R² {r2_score(actual, pred):6.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--predict", metavar="CSV")
    ap.add_argument("--out", default=paths.expl("best_v3_predictions.csv"))
    args = ap.parse_args()
    m.configure_cpu_parallelism()

    if args.fit or not _artifacts_exist():
        fit_and_save()

    target = args.predict or HOLDOUT
    mlp_total, xgb_total, blend = predict_parts(target)
    pd.DataFrame({"predictions": blend}).to_csv(args.out, index=False)
    print(f"wrote {len(blend)} predictions -> {args.out}")

    if args.predict is None:  # scored the mini holdout — full ablation + region breakdown
        ans = pd.read_csv(ANSWERS_URL)
        actual = (ans["casual"] + ans["registered"]).to_numpy()

        print("\n=== V3 (holiday-aware) on mini holdout ===")
        print(_score("V3 MLP only", mlp_total, actual))
        print(_score("V3 XGB only", xgb_total, actual))
        print(_score("V3 blend (MLP+XGB)  ★", blend, actual))

        # Reference: the saved V2 blend (no holiday features).
        try:
            import best_model
            v2 = best_model.predict(HOLDOUT)
            print(_score("V2 blend (baseline)", v2, actual))
        except Exception as e:  # noqa: BLE001
            v2 = None
            print(f"(V2 baseline unavailable: {e})")

        # Where did it change? Split the Thanksgiving neighbourhood from the rest.
        ho = engineer_v3(HOLDOUT)
        thx = ho["dteday"].dt.normalize().between("2023-11-20", "2023-11-24").to_numpy()
        print("\n--- RMSE by region ---")
        print(f"{'region':22s} {'n':>4} {'V3 blend':>9}" + ("  V2 blend" if v2 is not None else ""))
        for label, mask in (("Thanksgiving wk 20-24", thx), ("rest of holdout", ~thx)):
            row = f"{label:22s} {mask.sum():4d} {root_mean_squared_error(actual[mask], blend[mask]):9.2f}"
            if v2 is not None:
                row += f" {root_mean_squared_error(actual[mask], v2[mask]):9.2f}"
            print(row)


if __name__ == "__main__":
    main()
