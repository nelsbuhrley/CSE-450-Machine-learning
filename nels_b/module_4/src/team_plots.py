"""Stakeholder-facing plots for the bike-rental advisory deck.

Five plots, one message each:

  1. 01_actual_vs_predicted_hourly.png — hourly actual vs V3-predicted on the
     Nov 15-30, 2023 blind-test window.
  2. 02_within_tolerance.png — share of forecasts that land within 10/20/30/50%
     of actual, both hourly and daily.
  3. 03_monthly_totals_history.png — actual vs predicted monthly total rentals,
     Jan 2011 – Dec 2023 (Nov/Dec 2023 = prediction-only tail).  Line chart.
  4. 04_covid_era_accuracy.png — same monthly view, with a "no-COVID" counter-
     factual trend fit on 2011 – Feb 2020 data, projected forward.
  5. 05_sp500_overlay.png — bike-rental growth vs S&P 500 growth (both indexed
     to Jan 2011 = 100), pre-COVID and post-COVID recovery comparison.

All save to plots/team/ at dpi 150.  Predictions over the 13-year history are
cached to artifacts/v3_history_predictions.npz on first run.  S&P 500 monthly
closes live in data/sp500_monthly_2011_2023.csv.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import r2_score, root_mean_squared_error

import best_model_v3 as v3
import module4 as m
import paths

OUT = os.path.join(paths.PLOTS, "team")
HISTORY_CSV = os.path.join(paths.DATA, "bikes.csv")
SP500_CSV = os.path.join(paths.DATA, "sp500_monthly_2011_2023.csv")
PRED_CACHE = os.path.join(paths.ARTIFACTS, "v3_history_predictions.npz")
NOV_DEC_CSV = paths.BIG_HOLDOUT  # bikes_december.csv = Nov 1 – Dec 31, 2023 features

# Color system — kept tight so the four plots feel like a set.
C_ACTUAL = "#222222"
C_PRED = "#1f77b4"
C_FILL = "#1f77b4"
C_GOOD = "#2ca02c"
C_BAD = "#d62728"
C_NEUTRAL = "#7f7f7f"

# COVID era cutpoints (DC bike-share saw the operational shock from March 2020;
# ridership was meaningfully recovering by mid-2021).
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-06-30")


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #

def _load_history():
    """Full bikes.csv with actuals + V3 predictions, plus a `dt` timestamp."""
    df = pd.read_csv(HISTORY_CSV)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["dt"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df["actual"] = df["casual"] + df["registered"]

    if os.path.exists(PRED_CACHE):
        df["pred"] = np.load(PRED_CACHE)["pred"]
    else:
        print(f"running V3 predict over {len(df):,} historical rows (one-time, ~60s)…")
        pred = v3.predict(HISTORY_CSV)
        np.savez(PRED_CACHE, pred=pred)
        df["pred"] = pred
        print(f"cached predictions -> {PRED_CACHE}")
    return df


def _load_holdout():
    """Nov 1 - Dec 31, 2023 features, V3 predictions, and Nov 15-30 actuals overlay."""
    df = pd.read_csv(paths.BIG_HOLDOUT)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["dt"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df["pred"] = v3.predict(paths.BIG_HOLDOUT)

    ans = pd.read_csv(paths.ANSWERS_URL)
    mini = pd.read_csv(paths.MINI_HOLDOUT)
    mini["dteday"] = pd.to_datetime(mini["dteday"])
    mini["actual"] = ans["casual"] + ans["registered"]
    df = df.merge(mini[["dteday", "hr", "actual"]], on=["dteday", "hr"], how="left")
    return df


def _monthly_series():
    """Predicted vs actual monthly totals, extended to Dec 2023 with a forecast-only tail.

    Returns a DataFrame indexed by month start with columns:
        actual, pred, kind  ('history' for in-sample, 'forecast' for Nov-Dec 2023).
    """
    hist = _load_history()
    mo = _monthly(hist).reset_index()
    mo["kind"] = "history"

    # Forecast-only extension: Nov + Dec 2023 from bikes_december.csv (no actuals).
    nd = pd.read_csv(NOV_DEC_CSV)
    nd["dteday"] = pd.to_datetime(nd["dteday"])
    nd["pred"] = v3.predict(NOV_DEC_CSV)
    nd["month"] = nd["dteday"].dt.to_period("M").dt.to_timestamp()
    nd_mo = nd.groupby("month")["pred"].sum().reset_index()
    nd_mo["actual"] = np.nan
    nd_mo["kind"] = "forecast"

    out = pd.concat([mo, nd_mo[["month", "actual", "pred", "kind"]]],
                    ignore_index=True).sort_values("month").reset_index(drop=True)
    return out


def _load_sp500():
    """S&P 500 monthly closes, Jan 2011 – Dec 2023."""
    df = pd.read_csv(SP500_CSV, parse_dates=["month"])
    return df


def _no_covid_counterfactual(mo):
    """Fit a trend + seasonality on pre-COVID actuals; project monthly totals forward.

    Features: year_idx (linear time) + 12-month Fourier (sin/cos, 2 harmonics).
    Fit on actuals from Jan 2011 – Feb 2020.  Predict over the full mo.month range.
    Returns a Series aligned to mo["month"].
    """
    from sklearn.linear_model import LinearRegression

    def feats(months):
        t = (months - pd.Timestamp("2011-01-01")).dt.days.values / 365.25
        m_of_y = months.dt.month.values
        ang = 2 * np.pi * m_of_y / 12
        return np.column_stack([t, np.sin(ang), np.cos(ang),
                                np.sin(2 * ang), np.cos(2 * ang)])

    pre = mo[(mo["month"] < COVID_START) & mo["actual"].notna()]
    X_pre = feats(pre["month"])
    reg = LinearRegression().fit(X_pre, pre["actual"].values)
    X_all = feats(mo["month"])
    return pd.Series(reg.predict(X_all), index=mo.index)


# --------------------------------------------------------------------------- #
# plot helpers                                                                #
# --------------------------------------------------------------------------- #

def _pct_within(actual, pred, tol):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    err = np.abs(a - p) / np.clip(a, 1.0, None)
    return float((err <= tol).mean() * 100)


def _header(fig, title, subtitle, x=0.045, y_title=0.955, y_sub=0.905):
    """Two-line figure header with reliable spacing (no title/subtitle collision)."""
    fig.text(x, y_title, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(x, y_sub, subtitle, fontsize=10.5, color=C_NEUTRAL, ha="left", va="top")


def _save(name):
    out = os.path.join(OUT, name)
    plt.savefig(out, dpi=150, facecolor="white")
    plt.close()
    print(f"  -> {out}")


# --------------------------------------------------------------------------- #
# Plot 1 — hourly actual vs predicted, Nov 15-30 holdout                      #
# --------------------------------------------------------------------------- #

def fig_actual_vs_predicted(df):
    gt = df[df["actual"].notna()].sort_values("dt")
    rmse = root_mean_squared_error(gt["actual"], gt["pred"])
    r2 = r2_score(gt["actual"], gt["pred"])
    within20 = _pct_within(gt["actual"], gt["pred"], 0.20)

    fig, ax = plt.subplots(figsize=(15, 6.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.12)
    ax.fill_between(gt["dt"], gt["actual"], gt["pred"],
                    color=C_FILL, alpha=0.12, linewidth=0)
    ax.plot(gt["dt"], gt["actual"], color=C_ACTUAL, lw=1.8, label="Actual rentals")
    ax.plot(gt["dt"], gt["pred"], color=C_PRED, lw=1.4, label="Model prediction")

    # Light shade for the Thanksgiving holiday week — the hardest stretch.
    thx_lo = pd.Timestamp("2023-11-20")
    thx_hi = pd.Timestamp("2023-11-25")
    ax.axvspan(thx_lo, thx_hi, color="#fde9d9", alpha=0.6, zorder=0)
    y_top = max(gt["actual"].max(), gt["pred"].max()) * 1.08
    ax.text(thx_lo + (thx_hi - thx_lo) / 2, y_top * 0.96,
            "Thanksgiving week", ha="center", va="top",
            fontsize=9, color="#a05a2c", style="italic")

    _header(fig,
            "Predicted hourly bike rentals closely track actual demand",
            f"Nov 15 – 30, 2023 blind-test window  ·  "
            f"R² {r2:.2f}  ·  RMSE {rmse:.0f} rentals / hr  ·  "
            f"{within20:.0f}% of hours within 20% of actual")
    ax.set_xlabel("")
    ax.set_ylabel("Rentals per hour")
    ax.set_ylim(0, y_top)
    ax.set_xlim(gt["dt"].min(), gt["dt"].max())
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%b %d"))
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.35)
    _save("01_actual_vs_predicted_hourly.png")
    return rmse, r2, within20


# --------------------------------------------------------------------------- #
# Plot 2 — within-tolerance accuracy                                          #
# --------------------------------------------------------------------------- #

def fig_within_tolerance(df):
    gt = df[df["actual"].notna()].copy()
    tols = [0.10, 0.20, 0.30, 0.50]
    labels = ["±10%", "±20%", "±30%", "±50%"]

    # Hourly hit rate.
    hourly = [_pct_within(gt["actual"], gt["pred"], t) for t in tols]
    # Daily total hit rate.
    by_day = gt.groupby(gt["dt"].dt.normalize()).agg(
        actual=("actual", "sum"), pred=("pred", "sum")
    )
    daily = [_pct_within(by_day["actual"], by_day["pred"], t) for t in tols]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.10, wspace=0.08)
    x = np.arange(len(tols))
    for ax, vals, title, sub in (
        (axL, hourly, "Hourly forecasts",
         f"{len(gt):,} hours in the Nov 15-30 window"),
        (axR, daily, "Daily-total forecasts",
         f"{len(by_day)} full days in the window"),
    ):
        colors = [C_GOOD if v >= 70 else C_PRED if v >= 50 else C_BAD for v in vals]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color=b.get_facecolor())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlabel("Tolerance band around actual")
        # Panel title + small caption, both inside the axes area for clarity.
        ax.text(0.02, 0.97, title, transform=ax.transAxes,
                fontsize=12.5, fontweight="bold", va="top", ha="left")
        ax.text(0.02, 0.92, sub, transform=ax.transAxes,
                fontsize=10, color=C_NEUTRAL, va="top", ha="left")
        ax.set_ylim(0, 110)
        ax.grid(True, axis="y", alpha=0.35)
    axL.set_ylabel("% of forecasts inside the band")

    _header(fig,
            "How often the model lands within tolerance",
            "Daily aggregates are more accurate than single-hour estimates — "
            "operations can plan around them with high confidence.",
            y_title=0.94, y_sub=0.89)
    _save("02_within_tolerance.png")
    return hourly, daily


# --------------------------------------------------------------------------- #
# Plot 3 — monthly totals over full history                                   #
# --------------------------------------------------------------------------- #

def _monthly(df):
    g = df.copy()
    g["month"] = g["dt"].dt.to_period("M").dt.to_timestamp()
    mo = g.groupby("month").agg(actual=("actual", "sum"), pred=("pred", "sum"))
    return mo


def fig_monthly_history(mo):
    hist = mo[mo["kind"] == "history"]
    abs_err_pct = (hist["pred"] - hist["actual"]).abs() / hist["actual"] * 100
    r2 = r2_score(hist["actual"], hist["pred"])
    mape = abs_err_pct.mean()
    within10 = (abs_err_pct <= 10).mean() * 100
    forecast_cut = hist["month"].max()  # last month with an actual

    fig, ax = plt.subplots(figsize=(15, 6.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.10)

    # Light shade over the forecast-only tail (Nov + Dec 2023).
    fc = mo[mo["kind"] == "forecast"]
    if len(fc):
        ax.axvspan(forecast_cut + pd.Timedelta(days=15),
                   fc["month"].max() + pd.Timedelta(days=15),
                   color="0.85", alpha=0.45, zorder=0)
        ax.text(fc["month"].iloc[0] + pd.Timedelta(days=15),
                mo["pred"].max() * 1.04,
                "forecast only\n(no actual yet)",
                ha="left", va="top", fontsize=9, color=C_NEUTRAL, style="italic")

    ax.plot(hist["month"], hist["actual"], color=C_ACTUAL, lw=2.0,
            marker="o", ms=4, label="Actual monthly rentals")
    ax.plot(mo["month"], mo["pred"], color=C_PRED, lw=1.7, ls="--",
            marker="s", ms=3.5, label="Predicted monthly rentals")
    ax.fill_between(hist["month"], hist["actual"], hist["pred"],
                    color=C_PRED, alpha=0.10, linewidth=0)

    _header(fig,
            "Model recovers the long-run trend and seasonality of the business",
            f"Monthly totals, Jan 2011 – Dec 2023  ·  R² {r2:.3f}  ·  "
            f"average monthly error {mape:.1f}%  ·  "
            f"{within10:.0f}% of months within 10%")
    ax.set_xlabel("")
    ax.set_ylabel("Total rentals per month")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.set_ylim(0, mo["pred"].max() * 1.12)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.35)
    _save("03_monthly_totals_history.png")
    return r2, mape, within10


# --------------------------------------------------------------------------- #
# Plot 4 — COVID-era accuracy                                                 #
# --------------------------------------------------------------------------- #

def _era(ts):
    if ts < COVID_START:
        return "Pre-COVID"
    if ts <= COVID_END:
        return "COVID disruption"
    return "Recovery"


def fig_covid_era(mo):
    mo = mo.copy()
    mo["era"] = mo["month"].apply(_era)
    mo["counterfactual"] = _no_covid_counterfactual(mo)
    hist = mo[mo["kind"] == "history"]
    palette = {"Pre-COVID": C_NEUTRAL, "COVID disruption": C_BAD, "Recovery": C_GOOD}

    fig, ax = plt.subplots(figsize=(15, 6.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.10)
    ax.axvspan(COVID_START, COVID_END, color=C_BAD, alpha=0.08, zorder=0)

    # No-COVID counterfactual: solid where it was fit (pre-COVID), dashed where projected.
    fit_mask = mo["month"] < COVID_START
    ax.plot(mo.loc[fit_mask, "month"], mo.loc[fit_mask, "counterfactual"],
            color="#888", lw=1.4, alpha=0.75)
    ax.plot(mo.loc[~fit_mask, "month"], mo.loc[~fit_mask, "counterfactual"],
            color="#888", lw=1.6, ls=(0, (4, 3)), alpha=0.85,
            label='"No-COVID" projection (fit on 2011 – Feb 2020)')

    ax.plot(hist["month"], hist["actual"], color=C_ACTUAL, lw=2.0,
            marker="o", ms=4, label="Actual monthly rentals")
    ax.plot(mo["month"], mo["pred"], color=C_PRED, lw=1.6, ls="--",
            marker="s", ms=3.5, label="Predicted monthly rentals")

    # Annotate the end-of-2023 gap between counterfactual and actual.
    last = hist.iloc[-1]
    cf_last = mo.loc[mo["month"] == last["month"], "counterfactual"].iloc[0]
    gap_pct = (last["actual"] - cf_last) / cf_last * 100
    ax.annotate(
        f"{'Behind' if gap_pct < 0 else 'Ahead of'} no-COVID path by "
        f"{abs(gap_pct):.0f}%\n({(last['actual'] - cf_last) / 1000:+.0f}K rentals / mo)",
        xy=(last["month"], last["actual"]),
        xytext=(-180, 60), textcoords="offset points",
        fontsize=10, ha="left", va="bottom", color="#444",
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.0,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="0.75", alpha=0.95),
    )

    # Era boundary labels at top.
    y_top = max(mo["actual"].max(), mo["pred"].max(),
                mo["counterfactual"].max()) * 1.18
    for era, txt_color in palette.items():
        sub = mo[mo["era"] == era]
        if sub.empty:
            continue
        mid = sub["month"].iloc[len(sub) // 2]
        ax.text(mid, y_top * 0.97, era, ha="center", va="top",
                fontsize=11, fontweight="bold", color=txt_color)

    # Per-era stats table.
    rows = []
    for era in ["Pre-COVID", "COVID disruption", "Recovery"]:
        sub = hist[hist["era"] == era]
        if sub.empty:
            continue
        err = (sub["pred"] - sub["actual"]).abs() / sub["actual"] * 100
        rows.append((era, len(sub), err.mean(),
                     r2_score(sub["actual"], sub["pred"])))
    txt_lines = ["Era             months   avg err   R²"]
    for era, n, e, r in rows:
        txt_lines.append(f"{era:<15} {n:>6}    {e:5.1f}%   {r:5.2f}")
    ax.text(0.012, 0.97, "\n".join(txt_lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.7", alpha=0.92))

    _header(fig,
            "COVID dip, recovery, and where the business would have been without it",
            "Monthly totals, 2011 – 2023  ·  shaded = Mar 2020 to Jun 2021 disruption  ·  "
            'gray = trend + seasonality fit on pre-COVID actuals, projected forward')
    ax.set_xlabel("")
    ax.set_ylabel("Total rentals per month")
    ax.set_ylim(0, y_top)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.35)
    _save("04_covid_era_accuracy.png")
    return rows, gap_pct


# --------------------------------------------------------------------------- #
# Plot 5 — bike rental growth vs S&P 500 growth                              #
# --------------------------------------------------------------------------- #

def fig_sp500_overlay(mo):
    """Both series indexed to Jan 2011 = 100, on a single y-axis for easy comparison."""
    sp = _load_sp500()
    hist = mo[mo["kind"] == "history"][["month", "actual"]].rename(
        columns={"actual": "bike"})
    df = hist.merge(sp, on="month", how="inner").dropna()

    # Smooth out the heavy monthly seasonality with a centered 12-month rolling mean.
    df = df.sort_values("month").reset_index(drop=True)
    df["bike_smooth"] = df["bike"].rolling(12, center=True, min_periods=6).mean()
    df["sp_smooth"] = df["sp500_close"].rolling(3, center=True, min_periods=1).mean()

    base = df.iloc[0]
    df["bike_idx"] = df["bike_smooth"] / base["bike_smooth"] * 100
    df["sp_idx"] = df["sp_smooth"] / base["sp500_close"] * 100

    # Era-bracket growth rates (CAGR) to call out in the legend.
    def cagr(sub, col):
        n_years = (sub["month"].iloc[-1] - sub["month"].iloc[0]).days / 365.25
        if n_years <= 0:
            return float("nan")
        return ((sub[col].iloc[-1] / sub[col].iloc[0]) ** (1 / n_years) - 1) * 100

    pre = df[df["month"] < COVID_START]
    post = df[df["month"] > COVID_END]
    bike_pre_cagr = cagr(pre, "bike_idx")
    sp_pre_cagr = cagr(pre, "sp_idx")
    bike_post_cagr = cagr(post, "bike_idx")
    sp_post_cagr = cagr(post, "sp_idx")

    fig, ax = plt.subplots(figsize=(15, 6.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.10)
    ax.axvspan(COVID_START, COVID_END, color=C_BAD, alpha=0.07, zorder=0)
    ax.axhline(100, color="0.7", lw=0.8, zorder=0)

    def _fmt(v):
        return f"{v:+.1f}%/yr"

    ax.plot(df["month"], df["bike_idx"], color=C_PRED, lw=2.2,
            label=f"Bike rentals (12-mo smoothed)   "
                  f"pre-COVID {_fmt(bike_pre_cagr)}  ·  "
                  f"recovery {_fmt(bike_post_cagr)}")
    ax.plot(df["month"], df["sp_idx"], color=C_GOOD, lw=2.0, ls="--",
            label=f"S&P 500                            "
                  f"pre-COVID {_fmt(sp_pre_cagr)}  ·  "
                  f"recovery {_fmt(sp_post_cagr)}")

    # End-of-series value labels.
    last = df.iloc[-1]
    for col, color, ypad in (("bike_idx", C_PRED, 12),
                             ("sp_idx", C_GOOD, -16)):
        ax.annotate(f"{last[col]:.0f}", xy=(last["month"], last[col]),
                    xytext=(8, ypad), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=color, va="center")

    # COVID label centered on the shaded band.
    ax.text(COVID_START + (COVID_END - COVID_START) / 2,
            max(df["bike_idx"].max(), df["sp_idx"].max()) * 1.08,
            "COVID disruption", ha="center", va="top",
            fontsize=11, fontweight="bold", color=C_BAD)

    _header(fig,
            "Did the bike business keep pace with the broader market?",
            "Both series indexed to Jan 2011 = 100  ·  bike rentals = 12-month "
            "smoothed actuals  ·  S&P 500 = monthly close")
    ax.set_xlabel("")
    ax.set_ylabel("Index (Jan 2011 = 100)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.grid(True, alpha=0.35)
    _save("05_sp500_overlay.png")
    return dict(bike_pre=bike_pre_cagr, sp_pre=sp_pre_cagr,
                bike_post=bike_post_cagr, sp_post=sp_post_cagr,
                bike_end=float(last["bike_idx"]), sp_end=float(last["sp_idx"]))


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(OUT, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.5",
                                         "axes.linewidth": 0.8})
    m.configure_cpu_parallelism()

    print("loading holdout (Nov 1 - Dec 31, 2023)…")
    holdout = _load_holdout()
    print("loading historical data + V3 predictions (2011 - Oct 2023)…")
    mo = _monthly_series()

    print("plotting…")
    rmse, r2, w20 = fig_actual_vs_predicted(holdout)
    h_hit, d_hit = fig_within_tolerance(holdout)
    mo_r2, mo_mape, mo_w10 = fig_monthly_history(mo)
    era_stats, gap_pct = fig_covid_era(mo)
    sp_stats = fig_sp500_overlay(mo)

    print("\n=== headline numbers ===")
    print(f"Nov 15-30 hourly:  RMSE {rmse:.1f}  R² {r2:.3f}  within-20% {w20:.0f}%")
    print(f"Within tolerance hourly  (10/20/30/50%): "
          f"{h_hit[0]:.0f} / {h_hit[1]:.0f} / {h_hit[2]:.0f} / {h_hit[3]:.0f}")
    print(f"Within tolerance daily   (10/20/30/50%): "
          f"{d_hit[0]:.0f} / {d_hit[1]:.0f} / {d_hit[2]:.0f} / {d_hit[3]:.0f}")
    print(f"Monthly 2011-2023:   R² {mo_r2:.3f}  MAPE {mo_mape:.1f}%  "
          f"within-10% {mo_w10:.0f}%")
    for era, n, e, r in era_stats:
        print(f"  {era:<18} n={n:<3}  avg err {e:5.2f}%   R² {r:5.3f}")
    print(f"End of 2023 vs no-COVID counterfactual: {gap_pct:+.1f}%")
    print(f"S&P overlay:  pre-COVID  bike +{sp_stats['bike_pre']:.1f}%/yr  "
          f"S&P +{sp_stats['sp_pre']:.1f}%/yr")
    print(f"              recovery   bike +{sp_stats['bike_post']:.1f}%/yr  "
          f"S&P +{sp_stats['sp_post']:.1f}%/yr")
    print(f"              end-2023   bike idx {sp_stats['bike_end']:.0f}  "
          f"S&P idx {sp_stats['sp_end']:.0f}")


if __name__ == "__main__":
    main()
