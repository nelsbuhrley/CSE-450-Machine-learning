"""Add an XGBoost base learner (different inductive bias → decorrelated errors,
strong on the sharp rush-hour peaks where the MLPs are weak), then stack it with
V2_best (and V1_best). Pushes through: XGB alone → linear stack → time-aware meta
that can route by hour. Goal: beat the single best model (V2 model1_b256, RMSE 136.74).
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import module4 as m
import paths
import v1_pipeline as v1
from grade_holdout import ANSWERS_URL, PLOT_DIR, grade, report_and_plot

V1_BEST = os.path.join(paths.MODELS_V1, "model1_b256.keras")
V2_BEST = os.path.join(paths.MODELS_V2, "model1_b256.keras")
HOLDOUT = paths.MINI_HOLDOUT
ROUTE_COLS = ["hour_sin", "hour_cos", "workingday"]

XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=50,
    n_jobs=8,
    random_state=m.RANDOM_STATE,
)


def train_xgb(X_tr, y_tr):
    """Two XGB regressors (casual, registered) with early stopping on a val slice."""
    v = int(len(X_tr) * 0.15)
    xtr, xval = X_tr.iloc[v:], X_tr.iloc[:v]
    models = {}
    for col in ("casual", "registered"):
        ytr, yval = y_tr[col].iloc[v:], y_tr[col].iloc[:v]
        reg = xgb.XGBRegressor(**XGB_PARAMS)
        reg.fit(xtr, ytr, eval_set=[(xval, yval)], verbose=False)
        models[col] = reg
    return models


def xgb_total(models, X):
    pred = sum(models[c].predict(X) for c in ("casual", "registered"))
    return np.clip(np.round(pred, 1), 0, None)


def mlp_total(model, X, y_scaler, invert):
    return invert(y_scaler, model.predict(X, verbose=0)).sum(axis=1)


def main():
    m.configure_cpu_parallelism()
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set(rc={"figure.figsize": (11.7, 8.27)})

    answers = pd.read_csv(ANSWERS_URL)
    actual = (answers["casual"] + answers["registered"]).to_numpy()

    # --- V2 features (raw engineered); one split reused for trees and (scaled) for the MLP ---
    data = m.load_and_engineer_features(m.DATA_URL)
    feat_cols = [c for c in data.columns if c not in ("dteday", "hr", "dow", "casual", "registered")]
    X = data[feat_cols]
    y = data[["casual", "registered"]]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=m.RANDOM_STATE)
    y_test_total = y_te.sum(axis=1).to_numpy()

    x_scaler = MinMaxScaler().fit(X_tr)
    y_scaler_v2 = MinMaxScaler().fit(np.log1p(y_tr))

    ho = m.load_and_engineer_features(HOLDOUT)
    Xho = ho[feat_cols]

    # --- XGBoost base learner ---
    print("Training XGBoost (casual + registered)...")
    xgbm = train_xgb(X_tr, y_tr)
    print(f"  best_iteration: casual={xgbm['casual'].best_iteration} "
          f"registered={xgbm['registered'].best_iteration}")
    xgb_test = xgb_total(xgbm, X_te)
    xgb_hold = xgb_total(xgbm, Xho)

    # --- V2 MLP best ---
    mdl_v2 = tf.keras.models.load_model(V2_BEST, compile=False)
    v2_test = mlp_total(mdl_v2, x_scaler.transform(X_te), y_scaler_v2, m.invert_targets)
    v2_hold = mlp_total(mdl_v2, x_scaler.transform(Xho), y_scaler_v2, m.invert_targets)

    # --- V1 MLP best (own pipeline) ---
    xs1, ys1, X_te_v1, y_te_v1, ns1 = v1.prepare_v1()
    assert np.array_equal(y_te_v1.sum(axis=1).to_numpy(), y_test_total), "split mismatch"
    mdl_v1 = tf.keras.models.load_model(V1_BEST, compile=False)
    v1_test = mlp_total(mdl_v1, X_te_v1, ys1, v1.invert_v1)
    v1_hold = mlp_total(mdl_v1, v1.holdout_features_v1(xs1, ns1), ys1, v1.invert_v1)

    # --- Diagnostics: is XGB actually stronger where the MLP is weak? ---
    hr_test = data.loc[X_te.index, "hr"].to_numpy()
    rush = np.isin(hr_test, [7, 8, 9, 16, 17, 18, 19])
    print("\nWeak-area check — RMSE on X_test rush hours (7-9,16-19) vs rest:")
    for nm, p in (("V2_MLP", v2_test), ("XGBoost", xgb_test)):
        print(f"  {nm:8s} rush={root_mean_squared_error(y_test_total[rush], p[rush]):7.2f}  "
              f"rest={root_mean_squared_error(y_test_total[~rush], p[~rush]):7.2f}")
    eV2, eXGB = v2_test - y_test_total, xgb_test - y_test_total
    print(f"\nError correlation V2 vs XGB on X_test: {np.corrcoef(eV2, eXGB)[0, 1]:.3f} "
          f"(was 0.84 for V1 vs V2)")

    rows = [
        grade("V2_best", v2_hold, actual),
        grade("XGB", xgb_hold, actual),
    ]
    for name, p in (("XGB", xgb_hold), ("combo_avg_V2_XGB", None)):
        pd.DataFrame({"predictions": p if p is not None else (v2_hold + xgb_hold) / 2}).to_csv(
            paths.expl(f"{name}-predictions.csv"), index=False
        )

    avg = np.clip(np.round((v2_hold + xgb_hold) / 2, 1), 0, None)
    rows.append(grade("combo_avg_V2_XGB", avg, actual))

    # --- Linear stacks (fit on X_test base preds, leak-free w.r.t. base learners) ---
    stacks = {
        "stack_V2_XGB": [v2_test, xgb_test],
        "stack_V1_V2_XGB": [v1_test, v2_test, xgb_test],
    }
    stack_holds = {
        "stack_V2_XGB": [v2_hold, xgb_hold],
        "stack_V1_V2_XGB": [v1_hold, v2_hold, xgb_hold],
    }
    lin_coefs = {}
    for name, feats in stacks.items():
        est = LinearRegression(positive=True).fit(np.column_stack(feats), y_test_total)
        lin_coefs[name] = est
        sp = np.clip(np.round(est.predict(np.column_stack(stack_holds[name])), 1), 0, None)
        rows.append(grade(name, sp, actual))
        pd.DataFrame({"predictions": sp}).to_csv(paths.expl(f"{name}-predictions.csv"), index=False)

    # --- Time-aware meta: XGB meta on base preds + hour/workingday → can route by hour ---
    route_test = data.loc[X_te.index, ROUTE_COLS].to_numpy()
    route_hold = ho[ROUTE_COLS].to_numpy()
    meta_train = np.column_stack([v1_test, v2_test, xgb_test, route_test])
    meta_hold = np.column_stack([v1_hold, v2_hold, xgb_hold, route_hold])
    meta = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
        colsample_bytree=0.9, reg_lambda=1.0, objective="reg:squarederror", n_jobs=8,
        random_state=m.RANDOM_STATE,
    ).fit(meta_train, y_test_total)
    routed = np.clip(np.round(meta.predict(meta_hold), 1), 0, None)
    rows.append(grade("meta_routed", routed, actual))
    pd.DataFrame({"predictions": routed}).to_csv(paths.expl("meta_routed-predictions.csv"), index=False)

    # Cache base predictions so the meta-model can be tuned without retraining.
    np.savez(
        paths.BASE_PREDS,
        v1_test=v1_test, v2_test=v2_test, xgb_test=xgb_test,
        v1_hold=v1_hold, v2_hold=v2_hold, xgb_hold=xgb_hold,
        y_test_total=y_test_total, actual=actual,
        route_test=route_test, route_hold=route_hold,
    )

    for r in rows:
        report_and_plot(r)

    print("\nLinear stack weights (non-negative):")
    for name, est in lin_coefs.items():
        print(f"  {name:16s} intercept={est.intercept_:+7.3f}  coefs={np.round(est.coef_, 3)}")

    df = (
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        .set_index("model")
        .round(3)
        .sort_values("RMSE")
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== XGB + stacking grading (mini holdout, sorted by RMSE) ===")
    print(df.to_string())


if __name__ == "__main__":
    main()
