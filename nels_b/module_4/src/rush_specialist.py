"""Third base: a rush-hour specialist gated to working-day commute hours.

Trained ONLY on working-day rush-hour rows (7-9, 16-19), and applied ONLY on those
rows at inference. On non-work days it never fires, so by construction it cannot
create a rush-hour peak there — the blend falls back to V2+XGB.

Final (on the mini holdout):
  working-day rush rows -> mean(V2, XGB, specialist)
  all other rows        -> mean(V2, XGB)

Reuses cached V2/XGB predictions from base_preds.npz (run xgb_stack.py first).
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

import module4 as m
import paths
from grade_holdout import PLOT_DIR, grade, report_and_plot

HOLDOUT = paths.MINI_HOLDOUT
RUSH = [7, 8, 9, 16, 17, 18, 19]


def rush_workday_mask(hr, workingday):
    return np.isin(hr, RUSH) & (np.asarray(workingday) == 1)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    m.configure_cpu_parallelism()

    # Rebuild features + split (same seed → aligns with cached base preds).
    data = m.load_and_engineer_features(m.DATA_URL)
    feat_cols = [c for c in data.columns if c not in ("dteday", "hr", "dow", "casual", "registered")]
    X, y = data[feat_cols], data[["casual", "registered"]]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=m.RANDOM_STATE)
    ho = m.load_and_engineer_features(HOLDOUT)
    Xho = ho[feat_cols]

    d = np.load("base_preds.npz")
    assert np.array_equal(d["y_test_total"], y_te.sum(axis=1).to_numpy()), "split mismatch vs cache"
    v2_test, xgb_test = d["v2_test"], d["xgb_test"]
    v2_hold, xgb_hold = d["v2_hold"], d["xgb_hold"]
    y_test_total, actual = d["y_test_total"], d["actual"]

    # Masks: working-day rush hours.
    hr_tr = data.loc[X_tr.index, "hr"].to_numpy()
    hr_te = data.loc[X_te.index, "hr"].to_numpy()
    hr_ho = ho["hr"].to_numpy()
    mask_tr = rush_workday_mask(hr_tr, X_tr["workingday"])
    mask_te = rush_workday_mask(hr_te, X_te["workingday"])
    mask_ho = rush_workday_mask(hr_ho, Xho["workingday"])
    print(f"working-day rush rows — train {mask_tr.sum()}, test {mask_te.sum()}, "
          f"holdout {mask_ho.sum()}/{len(mask_ho)}")

    # Train specialist on working-day rush rows only (predict total).
    ytr_total = y_tr.sum(axis=1)
    Xs, ys = X_tr[mask_tr], ytr_total[mask_tr]
    v = int(len(Xs) * 0.15)
    spec = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
        objective="reg:squarederror", eval_metric="rmse", early_stopping_rounds=50,
        n_jobs=8, random_state=m.RANDOM_STATE,
    )
    spec.fit(Xs.iloc[v:], ys.iloc[v:], eval_set=[(Xs.iloc[:v], ys.iloc[:v])], verbose=False)
    print(f"specialist best_iteration={spec.best_iteration}")

    S_test = np.clip(np.round(spec.predict(X_te), 1), 0, None)
    S_hold = np.clip(np.round(spec.predict(Xho), 1), 0, None)

    # Gated combination.
    def combine(v2, xgb_, s, mask):
        out = (v2 + xgb_) / 2.0
        out = out.copy()
        out[mask] = (v2[mask] + xgb_[mask] + s[mask]) / 3.0
        return np.clip(np.round(out, 1), 0, None)

    blend2_hold = np.clip(np.round((v2_hold + xgb_hold) / 2.0, 1), 0, None)
    final_hold = combine(v2_hold, xgb_hold, S_hold, mask_ho)
    final_test = combine(v2_test, xgb_test, S_test, mask_te)
    blend2_test = np.clip(np.round((v2_test + xgb_test) / 2.0, 1), 0, None)

    # --- Constraint check: specialist must not touch non-work days ---
    nonwork_rush = np.isin(hr_ho, RUSH) & (Xho["workingday"].to_numpy() == 0)
    assert np.array_equal(final_hold[nonwork_rush], blend2_hold[nonwork_rush]), \
        "specialist leaked onto non-work-day rush hours"
    print("\nConstraint check — non-work-day rush hours (specialist excluded):")
    print(f"  rows={nonwork_rush.sum()}  mean actual={actual[nonwork_rush].mean():7.1f}  "
          f"mean final={final_hold[nonwork_rush].mean():7.1f}  "
          f"(specialist WOULD say {S_hold[nonwork_rush].mean():7.1f} — why we gate it)")
    print("Working-day rush hours (specialist active):")
    print(f"  rows={mask_ho.sum()}  mean actual={actual[mask_ho].mean():7.1f}  "
          f"mean blend2={blend2_hold[mask_ho].mean():7.1f}  mean final={final_hold[mask_ho].mean():7.1f}")

    # --- Peak-row RMSE: did the specialist help where it fires? ---
    print("\nHoldout RMSE on working-day rush rows:")
    for nm, p in (("blend2(V2,XGB)", blend2_hold), ("specialist alone", S_hold), ("final(3-base)", final_hold)):
        print(f"  {nm:18s} {root_mean_squared_error(actual[mask_ho], p[mask_ho]):7.2f}")

    pd.DataFrame({"predictions": final_hold}).to_csv("final_3base_rush-predictions.csv", index=False)
    pd.DataFrame({"predictions": S_hold}).to_csv("rush_specialist-predictions.csv", index=False)
    pd.DataFrame({"predictions": blend2_hold}).to_csv("blend2_V2_XGB-predictions.csv", index=False)

    rows = [
        grade("final_3base_rush", final_hold, actual),
        grade("blend2_V2_XGB", blend2_hold, actual),
    ]
    report_and_plot(rows[0])
    df = (
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        .set_index("model")
        .round(3)
    )
    pd.set_option("display.width", 200)
    print("\n=== Gated 3-base (V2 + XGB + rush specialist) vs 2-base blend (holdout) ===")
    print(df.to_string())


if __name__ == "__main__":
    main()
