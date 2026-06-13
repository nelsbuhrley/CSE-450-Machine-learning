"""Plots for the V3 holiday-aware best model, with a direct V2-vs-V3 comparison."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error

import best_model
import best_model_v3 as v3
import paths

OUT = os.path.join(paths.PLOTS, "v3_holiday")
ANSWERS_URL = paths.ANSWERS_URL
HOLDOUT = paths.MINI_HOLDOUT


def main():
    os.makedirs(OUT, exist_ok=True)
    sns.set_theme(style="darkgrid")

    ans = pd.read_csv(ANSWERS_URL)
    actual = (ans["casual"] + ans["registered"]).to_numpy()

    mlp_t, xgb_t, v3_blend = v3.predict_parts(HOLDOUT)
    v2_blend = best_model.predict(HOLDOUT)

    ho = v3.engineer_v3(HOLDOUT)
    dates = ho["dteday"].dt.normalize()
    thx = dates.between("2023-11-20", "2023-11-24").to_numpy()

    rmse = lambda p: root_mean_squared_error(actual, p)
    r2 = lambda p: r2_score(actual, p)

    # 1) Scatter: actual vs predicted, colored by 20% bucket.
    df = pd.DataFrame({"actual": actual, "predictions": v3_blend})
    df["pct"] = (df.actual - df.predictions).abs() / df.actual.clip(lower=1)
    df["bucket"] = np.where(df.pct >= 0.2, "off by >20%", "within 20%")
    plt.figure(figsize=(9, 8))
    ax = sns.scatterplot(data=df, x="actual", y="predictions", hue="bucket",
                         palette={"within 20%": "tab:blue", "off by >20%": "tab:orange"}, s=28)
    mx = max(df.actual.max(), df.predictions.max()) * 1.02
    ax.plot([0, mx], [0, mx], "r--", lw=1, label="perfect")
    ax.set(xlim=(0, mx), ylim=(0, mx), xlabel="Actual rentals", ylabel="Predicted rentals")
    within = (df.pct < 0.2).mean() * 100
    ax.set_title(f"V3 holiday-aware blend — actual vs predicted (mini holdout)\n"
                 f"RMSE {rmse(v3_blend):.1f}, R² {r2(v3_blend):.3f}, within-20% {within:.0f}%")
    plt.tight_layout()
    plt.savefig(f"{OUT}/v3_scatter.png", dpi=120)
    plt.close()

    # 2) THE comparison: V2 vs V3 vs actual across holdout hours, Thanksgiving shaded.
    x = np.arange(len(actual))
    plt.figure(figsize=(20, 6))
    plt.plot(x, actual, color="black", lw=1.6, label="actual")
    plt.plot(x, v2_blend, color="tab:red", lw=1.1, alpha=0.75,
             label=f"V2 blend (no holidays)  RMSE {rmse(v2_blend):.1f}")
    plt.plot(x, v3_blend, color="tab:green", lw=1.3,
             label=f"V3 holiday-aware blend  RMSE {rmse(v3_blend):.1f}")
    lo, hi = np.where(thx)[0].min(), np.where(thx)[0].max()
    plt.axvspan(lo, hi, color="gold", alpha=0.18, label="Thanksgiving wk (Nov 20-24)")
    plt.xlabel("Holdout hour index (Nov 15 → Nov 30, 2023)")
    plt.ylabel("Total rentals")
    plt.title("V2 vs V3 across the holdout — holiday features cut the Thanksgiving-week "
              "over-prediction (shaded): 201.6 → 159.8 RMSE")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/v2_vs_v3_line.png", dpi=120)
    plt.close()

    # 3) Component view: MLP / XGB / blend vs actual.
    plt.figure(figsize=(20, 6))
    plt.plot(x, actual, color="black", lw=1.6, label="actual")
    plt.plot(x, mlp_t, color="tab:blue", lw=0.9, alpha=0.7, label=f"V3 MLP  RMSE {rmse(mlp_t):.1f}")
    plt.plot(x, xgb_t, color="tab:purple", lw=0.9, alpha=0.7, label=f"V3 XGB  RMSE {rmse(xgb_t):.1f}")
    plt.plot(x, v3_blend, color="tab:green", lw=1.4, label=f"V3 blend  RMSE {rmse(v3_blend):.1f}")
    plt.axvspan(lo, hi, color="gold", alpha=0.18)
    plt.xlabel("Holdout hour index (Nov 15 → Nov 30, 2023)")
    plt.ylabel("Total rentals")
    plt.title("V3 base learners and their equal-weight blend across the holdout")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/v3_components_line.png", dpi=120)
    plt.close()

    # 4) Residual-by-hour: mean over/under prediction per hour, V2 vs V3.
    hr = ho["hr"].to_numpy()
    res = pd.DataFrame({"hr": hr, "v2": v2_blend - actual, "v3": v3_blend - actual})
    by = res.groupby("hr")[["v2", "v3"]].mean()
    plt.figure(figsize=(12, 6))
    w = 0.4
    plt.bar(by.index - w / 2, by.v2, w, color="tab:red", alpha=0.8, label="V2 blend")
    plt.bar(by.index + w / 2, by.v3, w, color="tab:green", alpha=0.85, label="V3 blend")
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("Hour of day")
    plt.ylabel("Mean (predicted − actual)  ·  + = over-predict")
    plt.title("Mean residual by hour — V3 pulls the over-predicted commute peaks toward zero")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/residual_by_hour.png", dpi=120)
    plt.close()

    print(f"wrote 4 plots to {OUT}/")
    for f in ("v3_scatter", "v2_vs_v3_line", "v3_components_line", "residual_by_hour"):
        print(f"  {OUT}/{f}.png")


if __name__ == "__main__":
    main()
