"""Regularized convex meta-model over [V2 keras, XGB]: weights >= 0, sum to 1,
shrunk toward equal. Tuned on cached X_test base predictions (run xgb_stack.py first
to produce base_preds.npz), graded on the mini holdout.

Objective per lambda:  mean((A w - y)^2) + lambda * ||w - uniform||^2
  lambda=0   -> pure least-squares blend (leans to whichever wins on X_test)
  lambda->inf -> equal weights (the simple average)
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

import paths
from grade_holdout import PLOT_DIR, grade, report_and_plot

BASES = ["v2", "xgb"]  # the keras one + xgboost
LAMBDAS = [0, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]


def fit_blend(A, y, lam):
    k = A.shape[1]
    u = np.full(k, 1.0 / k)

    def obj(w):
        r = A @ w - y
        return r @ r / len(y) + lam * ((w - u) @ (w - u))

    res = minimize(
        obj, u, method="SLSQP", bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return res.x


def cv_rmse(A, y, lam, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    errs = [
        root_mean_squared_error(y[va], A[va] @ fit_blend(A[tr], y[tr], lam))
        for tr, va in kf.split(A)
    ]
    return float(np.mean(errs))


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set(rc={"figure.figsize": (11.7, 8.27)})

    d = np.load(paths.BASE_PREDS)
    A_test = np.column_stack([d[f"{b}_test"] for b in BASES])
    A_hold = np.column_stack([d[f"{b}_hold"] for b in BASES])
    y, actual = d["y_test_total"], d["actual"]

    print(f"{'lambda':>9} {'w_v2':>6} {'w_xgb':>6} {'Xtest_CV':>9} {'holdout':>9}")
    rows_path = []
    for lam in LAMBDAS:
        w = fit_blend(A_test, y, lam)
        cv = cv_rmse(A_test, y, lam)
        hr = root_mean_squared_error(actual, np.clip(A_hold @ w, 0, None))
        rows_path.append((lam, w, cv, hr))
        print(f"{lam:9.0f} {w[0]:6.3f} {w[1]:6.3f} {cv:9.3f} {hr:9.3f}")

    # Holdout-blind selection. The unregularized (lambda=0) least-squares blend is what
    # a naive learned stacker would use. If it collapses onto a single base learner
    # (discards a comparably-good one), that's a red flag that the meta-training split
    # is unrepresentative -> distrust it and shrink fully to equal weights. Otherwise
    # keep the data-informed tilt.
    w0 = fit_blend(A_test, y, 0.0)
    collapsed = w0.min() < 0.2
    if collapsed:
        w_r = np.full(len(BASES), 1.0 / len(BASES))
        print(f"\nUnregularized blend collapsed to {np.round(w0, 3)} (discards a base "
              f"learner) -> shrink fully to equal weights {np.round(w_r, 3)}")
    else:
        w_r = w0
        print(f"\nUnregularized blend is balanced {np.round(w0, 3)} -> keep it")

    pred = np.clip(np.round(A_hold @ w_r, 1), 0, None)
    pd.DataFrame({"predictions": pred}).to_csv(paths.expl("meta_convex-predictions.csv"), index=False)

    rows = [
        grade("meta_convex", pred, actual),
        grade("avg_equal", np.clip(np.round(A_hold @ np.full(len(BASES), 1 / len(BASES)), 1), 0, None), actual),
        grade("v2_only", np.clip(np.round(A_hold[:, BASES.index("v2")], 1), 0, None), actual),
        grade("xgb_only", np.clip(np.round(A_hold[:, BASES.index("xgb")], 1), 0, None), actual),
    ]
    report_and_plot(rows[0])

    df = (
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        .set_index("model")
        .round(3)
    )
    pd.set_option("display.width", 200)
    print("\n=== Convex meta-model vs references (mini holdout) ===")
    print(df.to_string())


if __name__ == "__main__":
    main()
