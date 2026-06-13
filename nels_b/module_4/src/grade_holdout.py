"""Grade the 3 saved models on the mini holdout, then a 1/RMSE-weighted ensemble.

Reporting + visualizations mirror module04_biking_grading.ipynb: per-team metrics,
the actual-vs-predicted scatter (colored by 20% bucket), and the actual/predicted line
plot. Figures are written to plots/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

import module4 as m
import paths

PLOT_DIR = paths.PLOTS

ANSWERS_URL = paths.ANSWERS_URL
MODELS = [
    ("model1_b256", os.path.join(paths.MODELS_V2, "model1_b256.keras")),
    ("model2_b512", os.path.join(paths.MODELS_V2, "model2_b512.keras")),
    ("model3_b1024", os.path.join(paths.MODELS_V2, "model3_b1024.keras")),
]


def holdout_features(x_scaler):
    """Engineer the holdout and scale it with training's single x_scaler."""
    ho = m.load_and_engineer_features(paths.MINI_HOLDOUT)
    x = ho.drop(columns=["dteday", "hr", "dow"])
    return x_scaler.transform(x)


def total_predictions(model, X, y_scaler):
    """Predict scaled-log [casual, registered], invert, clip, return total per row."""
    return m.invert_targets(y_scaler, model.predict(X, verbose=0)).sum(axis=1)


def grade(name, pred_total, actual):
    rmse = root_mean_squared_error(actual, pred_total)
    mae = mean_absolute_error(actual, pred_total)
    medae = median_absolute_error(actual, pred_total)
    r2 = r2_score(actual, pred_total)
    absdiff_pct = np.abs(actual - pred_total) / actual
    within = {p: (absdiff_pct <= p / 100).mean() * 100 for p in (5, 10, 20)}

    # Per-row frame in the grading notebook's shape, used for the visualizations.
    testfinal = pd.DataFrame({"predictions": pred_total, "actual": actual})
    testfinal["difference"] = testfinal["actual"] - testfinal["predictions"]
    testfinal["percent_difference"] = (testfinal["difference"] / testfinal["actual"]).abs()
    testfinal["percent_bucket"] = np.where(
        testfinal["percent_difference"] >= 0.2, "above 20%", "below 20%"
    )

    return {
        "model": name,
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "R2": r2,
        "w/in 5%": within[5],
        "w/in 10%": within[10],
        "w/in 20%": within[20],
        "_within": within,
        "_testfinal": testfinal,
    }


def report_and_plot(result):
    """Print the notebook's per-team block and save its scatter + line plots."""
    name = result["model"]
    within = result["_within"]
    testfinal = result["_testfinal"]

    print(f"\n-------------------------------- {name.upper()} RESULTS ---------------------------------\n")
    print(
        f" Within 5%: {within[5]}%\n",
        f"Within 10%: {within[10]}%\n",
        f"Within 20%: {within[20]}%\n",
        f"R^2: {result['R2']}\n",
        f"RMSE: {result['RMSE']}\n",
        f"Mean Absolute Error: {result['MAE']}\n",
        f"Median Absolute Error: {result['MedAE']}",
    )

    # Scatter: actual vs predicted, colored by 20% bucket, with the perfect line.
    color_dict = {"below 20%": "tab:blue", "above 20%": "tab:orange"}
    plt.figure(figsize=(11.7, 8.27))
    ax = sns.scatterplot(
        data=testfinal, x="actual", y="predictions", hue="percent_bucket", palette=color_dict
    )
    xlims = (0, 1e3)
    ax.plot(xlims, xlims, color="r")
    ax.set_title(f"{name} — actual vs predicted")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_scatter.png", dpi=120)
    plt.close()

    # Line: actual and predicted across the holdout index.
    line = testfinal[["actual", "predictions"]].copy()
    line["i"] = line.index
    line = line.melt(["i"])
    plt.figure(figsize=(20, 6))
    sns.lineplot(data=line, x="i", y="value", hue="variable")
    plt.title(f"{name} — actual vs predicted over holdout")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_line.png", dpi=120)
    plt.close()


def main():
    m.configure_cpu_parallelism()
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set(rc={"figure.figsize": (11.7, 8.27)})

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Reconstruct the exact scaler used at train time (deterministic, RANDOM_STATE=42).
    X_train, X_test, y_test, y_scaler, y_train_scaled = m.prepare_data()
    data = m.load_and_engineer_features(m.DATA_URL)
    x_raw = data.drop(columns=["dteday", "hr", "dow", "casual", "registered"])

    y_raw = data[["casual", "registered"]]
    X_tr_raw, _, _, _ = train_test_split(
        x_raw, y_raw, test_size=0.3, random_state=m.RANDOM_STATE
    )
    x_scaler = MinMaxScaler().fit(X_tr_raw)

    answers = pd.read_csv(ANSWERS_URL)
    actual = (answers["casual"] + answers["registered"]).to_numpy()
    y_test_total = y_test.sum(axis=1).to_numpy()

    X_holdout = holdout_features(x_scaler)

    rows, preds, train_rmse, test_totals = [], [], [], []
    for name, path in MODELS:
        model = tf.keras.models.load_model(path, compile=False)

        # Base-model predictions on X_test (never seen in training): used both for the
        # 1/RMSE weights and as the meta-model's training features.
        tr_pred = m.invert_targets(y_scaler, model.predict(X_test, verbose=0))
        train_rmse.append(root_mean_squared_error(y_test, tr_pred))
        test_totals.append(tr_pred.sum(axis=1))

        pt = total_predictions(model, X_holdout, y_scaler)
        preds.append(pt)
        pd.DataFrame({"predictions": pt}).to_csv(paths.expl(f"{name}-predictions.csv"), index=False)
        rows.append(grade(name, pt, actual))

    # 1/RMSE-weighted ensemble of total predictions.
    w = np.array([1.0 / r for r in train_rmse])
    w /= w.sum()
    ens = np.clip(np.round(np.tensordot(w, np.stack(preds), axes=([0], [0])), 1), 0, None)
    pd.DataFrame({"predictions": ens}).to_csv(paths.expl("ensemble_weighted-predictions.csv"), index=False)
    rows.append(grade("ensemble_weighted", ens, actual))

    # Stacking meta-models: learn a blend of base totals on X_test, then apply to the
    # holdout. X_test was held out of base-model training, so the fit is leak-free w.r.t.
    # the base learners. Base models are collinear, so a plain OLS stacker overfits the
    # meta split; the constrained/regularized variants test whether that's fixable.
    meta_train = np.column_stack(test_totals)
    meta_holdout = np.column_stack(preds)
    metas = {
        "stack_ols": LinearRegression(),
        "stack_ridge": Ridge(alpha=100.0),
        "stack_nonneg": LinearRegression(positive=True),
    }
    fitted_metas = {}
    for mname, est in metas.items():
        est.fit(meta_train, y_test_total)
        fitted_metas[mname] = est
        sp = np.clip(np.round(est.predict(meta_holdout), 1), 0, None)
        pd.DataFrame({"predictions": sp}).to_csv(paths.expl(f"ensemble_{mname}-predictions.csv"), index=False)
        rows.append(grade(f"ensemble_{mname}", sp, actual))

    for r in rows:
        report_and_plot(r)

    df = (
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        .set_index("model")
        .round(3)
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nEnsemble weights (1/train-RMSE):")
    for (name, _), wi, tr in zip(MODELS, w, train_rmse):
        print(f"  {name:14s} train_RMSE={tr:7.3f}  weight={wi:.3f}")
    print("\nStacking meta-models (fit on base totals over X_test):")
    for mname, est in fitted_metas.items():
        coefs = "  ".join(f"{n}={c:+.3f}" for (n, _), c in zip(MODELS, est.coef_))
        print(f"  {mname:13s} intercept={est.intercept_:+7.3f}  {coefs}")
    print("\n=== Mini holdout grading (target = casual + registered) ===")
    print(df.to_string())
    print(f"\nPlots written to {PLOT_DIR}/ (scatter + line per model).")


if __name__ == "__main__":
    main()
