"""Regenerate every model's holdout plots with descriptive titles, filed into the
organized plots/ subfolders (base, model, ensemble, combo, stack, meta).

Each plot title names what the model is and its holdout RMSE / R^2. Predictions come
from the cached base_preds.npz (base learners) and the *-predictions.csv files.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error

import paths

PLOT_DIR = paths.PLOTS
ANSWERS_URL = paths.ANSWERS_URL

# key, subfolder, description, source: ("npz", name) or ("csv", path)
REGISTRY = [
    # --- base learners ---
    ("V1_best", "base", "V1 MLP (model1_b256) — original pipeline, raw-count targets", ("npz", "v1_hold")),
    ("V2_best", "base", "V2 MLP (model1_b256) — +day-of-week & hour×workingday, log1p targets", ("npz", "v2_hold")),
    ("XGB", "base", "XGBoost — tree learner, strongest at rush-hour peaks", ("npz", "xgb_hold")),
    ("rush_specialist", "base", "Rush-hour specialist XGB — trained only on working-day commute hours (ungated)", ("csv", "rush_specialist-predictions.csv")),
    # --- individual V2 MLPs ---
    ("model1_b256", "model", "V2 MLP — batch 256 (best single MLP)", ("csv", "model1_b256-predictions.csv")),
    ("model2_b512", "model", "V2 MLP — batch 512", ("csv", "model2_b512-predictions.csv")),
    ("model3_b1024", "model", "V2 MLP — batch 1024", ("csv", "model3_b1024-predictions.csv")),
    # --- V2-only ensembles ---
    ("ensemble_weighted", "ensemble", "V2 ensemble — 1/RMSE-weighted average of 3 MLPs", ("csv", "ensemble_weighted-predictions.csv")),
    ("ensemble_stacked", "ensemble", "V2 ensemble — OLS stack of 3 MLPs", ("csv", "ensemble_stacked-predictions.csv")),
    ("ensemble_stack_ols", "ensemble", "V2 ensemble — OLS stacking meta", ("csv", "ensemble_stack_ols-predictions.csv")),
    ("ensemble_stack_ridge", "ensemble", "V2 ensemble — Ridge stacking meta", ("csv", "ensemble_stack_ridge-predictions.csv")),
    ("ensemble_stack_nonneg", "ensemble", "V2 ensemble — non-negative stacking meta", ("csv", "ensemble_stack_nonneg-predictions.csv")),
    # --- V1+V2 combos ---
    ("combo_avg", "combo", "V1+V2 combo — simple average", ("csv", "combo_avg-predictions.csv")),
    ("combo_ols", "combo", "V1+V2 combo — OLS stack", ("csv", "combo_ols-predictions.csv")),
    ("combo_ridge", "combo", "V1+V2 combo — Ridge stack", ("csv", "combo_ridge-predictions.csv")),
    ("combo_nonneg", "combo", "V1+V2 combo — non-negative stack", ("csv", "combo_nonneg-predictions.csv")),
    ("combo_avg_V2_XGB", "combo", "V2 MLP + XGBoost — simple average (first blend to beat any single model)", ("csv", "combo_avg_V2_XGB-predictions.csv")),
    ("blend2_V2_XGB", "combo", "V2 MLP + XGBoost — equal-weight blend (current best)", ("csv", "blend2_V2_XGB-predictions.csv")),
    # --- linear stacks with XGB ---
    ("stack_V2_XGB", "stack", "V2 + XGB — non-negative linear stack (collapsed onto XGB)", ("csv", "stack_V2_XGB-predictions.csv")),
    ("stack_V1_V2_XGB", "stack", "V1 + V2 + XGB — non-negative linear stack", ("csv", "stack_V1_V2_XGB-predictions.csv")),
    # --- meta-models ---
    ("meta_convex", "meta", "V2 MLP + XGBoost — regularized convex meta (shrinks to equal weights) — BEST", ("csv", "meta_convex-predictions.csv")),
    ("meta_routed", "meta", "V1+V2+XGB — XGBoost routing meta (time-aware)", ("csv", "meta_routed-predictions.csv")),
    ("final_3base_rush", "meta", "V2 + XGB + gated rush specialist — specialist active only on working-day rush hours", ("csv", "final_3base_rush-predictions.csv")),
]


CSV_DIR = paths.EXPLORATION  # scratch *-predictions.csv live here


def load_pred(npz, source):
    kind, val = source
    if kind == "npz":
        return npz[val]
    return pd.read_csv(os.path.join(CSV_DIR, val))["predictions"].to_numpy()


def plot_model(key, folder, desc, pred, actual):
    rmse = root_mean_squared_error(actual, pred)
    r2 = r2_score(actual, pred)
    out_dir = os.path.join(PLOT_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({"actual": actual, "predictions": pred})
    df["pct"] = (df.actual - df.predictions).abs() / df.actual
    df["bucket"] = np.where(df.pct >= 0.2, "off by >20%", "within 20%")

    # Scatter: actual vs predicted, colored by 20% bucket.
    plt.figure(figsize=(11.7, 8.27))
    ax = sns.scatterplot(
        data=df, x="actual", y="predictions", hue="bucket",
        palette={"within 20%": "tab:blue", "off by >20%": "tab:orange"}, s=28,
    )
    mx = max(df.actual.max(), df.predictions.max()) * 1.02
    ax.plot([0, mx], [0, mx], "r--", lw=1, label="perfect")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("Actual rentals")
    ax.set_ylabel("Predicted rentals")
    ax.set_title(f"{desc}\nActual vs predicted on mini holdout  ·  RMSE {rmse:.1f}, R² {r2:.3f}",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{key}_scatter.png"), dpi=120)
    plt.close()

    # Line: actual & predicted across the holdout hours.
    line = df[["actual", "predictions"]].reset_index().melt("index")
    plt.figure(figsize=(20, 6))
    sns.lineplot(data=line, x="index", y="value", hue="variable",
                 palette={"actual": "black", "predictions": "tab:orange"})
    plt.xlabel("Holdout hour index (Nov 15 → Nov 30, 2023)")
    plt.ylabel("Rentals")
    plt.title(f"{desc}\nActual vs predicted across holdout hours  ·  RMSE {rmse:.1f}, R² {r2:.3f}",
              fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{key}_line.png"), dpi=120)
    plt.close()
    return rmse, r2


def main():
    sns.set_theme(style="darkgrid")
    answers = pd.read_csv(ANSWERS_URL)
    actual = (answers["casual"] + answers["registered"]).to_numpy()
    npz = np.load(paths.BASE_PREDS)

    # Clean up loose plots in the root that belong in subfolders.
    for f in os.listdir(PLOT_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(PLOT_DIR, f))

    print(f"{'model':24s} {'folder':9s} {'RMSE':>8} {'R²':>7}")
    results = []
    for key, folder, desc, source in REGISTRY:
        pred = load_pred(npz, source)
        rmse, r2 = plot_model(key, folder, desc, pred, actual)
        results.append((key, folder, rmse, r2))
        print(f"{key:24s} {folder:9s} {rmse:8.2f} {r2:7.3f}")

    # Summary: RMSE comparison across all models (best at top). Exclude the ungated
    # specialist — it's a gated component, not a standalone total predictor.
    res = sorted((r for r in results if r[0] != "rush_specialist"), key=lambda r: r[2])
    plt.figure(figsize=(11, 9))
    names = [r[0] for r in res]
    rmses = [r[2] for r in res]
    colors = ["tab:green" if r[2] == rmses[0] else "steelblue" for r in res]
    y = np.arange(len(names))[::-1]
    plt.barh(y, rmses, color=colors)
    plt.yticks(y, names)
    for yi, rm in zip(y, rmses):
        plt.text(rm + 0.3, yi, f"{rm:.1f}", va="center", fontsize=8)
    plt.xlabel("Holdout RMSE (lower is better)")
    plt.title("All models — holdout RMSE comparison\nGreen = best (equal-weight V2+XGB blend)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "_summary_rmse.png"), dpi=120)
    plt.close()
    print(f"\nWrote {len(results)} model plot-pairs + plots/_summary_rmse.png")


if __name__ == "__main__":
    main()
