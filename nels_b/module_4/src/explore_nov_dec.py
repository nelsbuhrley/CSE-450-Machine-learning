"""Thorough visual exploration of the model's Nov–Dec 2023 bike-demand predictions.

Predicted = V3 holiday-aware blend over bikes_december.csv (Nov 1 – Dec 31, 2023).
Actuals are overlaid for Nov 15–30 (the mini-holdout window — the only stretch with
ground truth). Weekends are shaded, holidays marked, with day-type hour profiles and
a day×hour heatmap. Outputs to plots/nov_dec/.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error

import best_model_v3 as v3
import paths
from make_submissions import v2_predict

OUT = os.path.join(paths.PLOTS, "nov_dec")

# Holidays + notable days in the window (date -> label).
HOLIDAYS = {
    "2023-11-10": "Veterans Day\n(observed)",
    "2023-11-23": "Thanksgiving",
    "2023-11-24": "Black Friday",
    "2023-12-25": "Christmas",
    "2023-12-31": "New Year's Eve",
}
RUSH = [7, 8, 9, 16, 17, 18, 19]


def load():
    df = pd.read_csv(paths.BIG_HOLDOUT)
    df["dt"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df["date"] = df["dt"].dt.normalize()
    df["dow"] = df["dt"].dt.dayofweek
    df["weekend"] = df["dow"] >= 5
    df["v3"] = v3.predict(paths.BIG_HOLDOUT)
    df["v2"] = v2_predict(paths.BIG_HOLDOUT)

    # Ground truth for the Nov 15-30 overlap.
    ans = pd.read_csv(paths.ANSWERS_URL)
    mini = pd.read_csv(paths.MINI_HOLDOUT)
    mini["actual"] = ans["casual"] + ans["registered"]
    df = df.merge(mini[["dteday", "hr", "actual"]], on=["dteday", "hr"], how="left")

    # Day type: holiday > weekend > weekday (holiday flag from the data).
    df["day_type"] = np.where(
        df["holiday"] == 1, "Holiday", np.where(df["weekend"], "Weekend", "Weekday")
    )
    return df


def shade_weekends(ax, df):
    for d in df.loc[df["weekend"], "date"].unique():
        d = pd.Timestamp(d)
        ax.axvspan(d, d + pd.Timedelta(days=1), color="0.85", alpha=0.55, zorder=0)


def mark_holidays(ax, ymax, label=True):
    for ds, name in HOLIDAYS.items():
        d = pd.Timestamp(ds) + pd.Timedelta(hours=12)
        ax.axvline(d, color="crimson", ls="--", lw=1.1, alpha=0.8, zorder=1)
        if label:
            ax.text(d, ymax * 0.98, name, rotation=90, va="top", ha="right",
                    fontsize=8, color="crimson", alpha=0.9)


def fig_timeline(df):
    fig, ax = plt.subplots(figsize=(22, 7))
    gt = df["actual"].notna()
    lo, hi = df.loc[gt, "dt"].min(), df.loc[gt, "dt"].max()
    ymax = max(df["v3"].max(), df["actual"].max()) * 1.08

    shade_weekends(ax, df)
    ax.axvspan(lo, hi, color="tab:green", alpha=0.06, zorder=0)
    ax.plot(df["dt"], df["v3"], color="tab:blue", lw=0.9, label="Predicted (V3 blend)")
    ax.plot(df.loc[gt, "dt"], df.loc[gt, "actual"], color="black", lw=1.1,
            label="Actual (Nov 15–30, known)")
    mark_holidays(ax, ymax)

    # Label the shaded helpers once.
    ax.axvspan(np.nan, np.nan, color="0.85", alpha=0.55, label="Weekend")
    ax.text(lo + (hi - lo) / 2, ymax * 0.92, "ground-truth window",
            ha="center", color="tab:green", fontsize=9)

    rmse = root_mean_squared_error(df.loc[gt, "actual"], df.loc[gt, "v3"])
    r2 = r2_score(df.loc[gt, "actual"], df.loc[gt, "v3"])
    ax.set_title(f"Hourly bike demand, Nov–Dec 2023 — V3 model prediction vs actual\n"
                 f"(on the known Nov 15–30 window: RMSE {rmse:.1f}, R² {r2:.3f})", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rentals / hour")
    ax.set_ylim(0, ymax)
    ax.set_xlim(df["dt"].min(), df["dt"].max())
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%a"))
    ax.legend(loc="upper left", ncol=4, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_timeline_hourly.png", dpi=120)
    plt.close()


def fig_daily(df):
    daily = df.groupby("date").agg(
        v3=("v3", "sum"), v2=("v2", "sum"), actual=("actual", "sum"),
        weekend=("weekend", "first"), holiday=("holiday", "max"),
        n=("v3", "size"),
    ).reset_index()
    daily.loc[daily["n"] < 24, "actual"] = np.nan  # only full days have a real total
    has_actual = daily.groupby("date")["actual"].first().notna()

    fig, ax = plt.subplots(figsize=(20, 6))
    shade_weekends(ax, df)
    colors = np.where(daily["holiday"] == 1, "crimson",
                      np.where(daily["weekend"], "tab:orange", "tab:blue"))
    ax.bar(daily["date"], daily["v3"], width=0.8, color=colors, alpha=0.75,
           label="Predicted daily total (V3)")
    ax.plot(daily["date"], daily["v2"], color="purple", lw=1.0, ls=":",
            marker="o", ms=3, label="Predicted (V2 pure-NN)")
    av = daily.dropna(subset=["actual"])
    ax.plot(av["date"], av["actual"], color="black", lw=1.6, marker="o", ms=5,
            label="Actual (Nov 15–30)")
    mark_holidays(ax, daily["v3"].max() * 1.05)

    # Color legend proxies.
    from matplotlib.patches import Patch
    handles = [Patch(color="tab:blue", label="Weekday (pred)"),
               Patch(color="tab:orange", label="Weekend (pred)"),
               Patch(color="crimson", label="Holiday (pred)")]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0][1:], loc="upper right", ncol=3)
    ax.set_title("Daily total rentals, Nov–Dec 2023 — predicted (bars, colored by day type) "
                 "vs actual where known", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rentals / day")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_daily_totals.png", dpi=120)
    plt.close()


def fig_profiles(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Predicted average hour-of-day curve by day type.
    palette = {"Weekday": "tab:blue", "Weekend": "tab:orange", "Holiday": "crimson"}
    prof = df.groupby(["day_type", "hr"])["v3"].mean().reset_index()
    for t, g in prof.groupby("day_type"):
        ax1.plot(g["hr"], g["v3"], marker="o", ms=3, color=palette[t], label=t)
    for h in RUSH:
        ax1.axvspan(h - 0.5, h + 0.5, color="0.9", zorder=0)
    ax1.set_title("Predicted average demand by hour of day\n"
                  "Weekday = bimodal commute peaks; Weekend/Holiday = single midday hump")
    ax1.set_xlabel("Hour of day  (shaded = rush hours)")
    ax1.set_ylabel("Mean predicted rentals")
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend(title="Day type")

    # Predicted vs actual hour profile on the ground-truth window.
    gt = df[df["actual"].notna()]
    pa = gt.groupby("hr").agg(pred=("v3", "mean"), actual=("actual", "mean")).reset_index()
    ax2.plot(pa["hr"], pa["actual"], color="black", marker="o", ms=4, label="Actual")
    ax2.plot(pa["hr"], pa["pred"], color="tab:blue", marker="o", ms=4, label="Predicted (V3)")
    ax2.fill_between(pa["hr"], pa["actual"], pa["pred"], color="tab:blue", alpha=0.12)
    for h in RUSH:
        ax2.axvspan(h - 0.5, h + 0.5, color="0.9", zorder=0)
    ax2.set_title("Predicted vs actual by hour (Nov 15–30 window)\n"
                  "Model tracks the commute peaks closely")
    ax2.set_xlabel("Hour of day  (shaded = rush hours)")
    ax2.set_ylabel("Mean rentals")
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_hour_profiles.png", dpi=120)
    plt.close()


def fig_heatmap(df):
    piv = df.pivot_table(index="date", columns="hr", values="v3")
    piv.index = [pd.Timestamp(d).strftime("%a %b %d") for d in piv.index]
    fig, ax = plt.subplots(figsize=(15, 16))
    sns.heatmap(piv, cmap="viridis", cbar_kws={"label": "Predicted rentals"}, ax=ax)
    ax.set_title("Predicted demand heatmap — every day × hour, Nov–Dec 2023\n"
                 "Bright vertical bands = commute peaks; dark rows = weekends/holidays",
                 fontsize=13)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_heatmap_day_hour.png", dpi=120)
    plt.close()


def fig_best_week(df):
    """Zoom on the cleanest-tracking 7-day window (lowest RMSE) inside the known range."""
    gt = df[df["actual"].notna()].sort_values("dt")
    days = sorted(pd.Timestamp(d) for d in gt["date"].unique())
    best = None
    for start in days:
        end = start + pd.Timedelta(days=7)
        w = gt[(gt["dt"] >= start) & (gt["dt"] < end)]
        if w["date"].nunique() < 7:
            continue
        rmse = root_mean_squared_error(w["actual"], w["v3"])
        if best is None or rmse < best[0]:
            best = (rmse, start, end, w)
    rmse, start, end, w = best
    r2 = r2_score(w["actual"], w["v3"])
    within = (np.abs(w["actual"] - w["v3"]) / w["actual"].clip(lower=1) <= 0.2).mean() * 100

    fig, ax = plt.subplots(figsize=(18, 6.5))
    shade_weekends(ax, w)
    # light rush-hour bands each day
    for d in pd.date_range(start, end - pd.Timedelta(hours=1), freq="D"):
        for h0, h1 in ((7, 9), (16, 19)):
            ax.axvspan(d + pd.Timedelta(hours=h0), d + pd.Timedelta(hours=h1),
                       color="0.92", zorder=0)

    ax.plot(w["dt"], w["actual"], color="black", lw=2.0, marker="o", ms=3.5, label="Actual")
    ax.plot(w["dt"], w["v3"], color="tab:blue", lw=1.8, label="Predicted (V3)")
    ax.fill_between(w["dt"], w["actual"], w["v3"], color="tab:blue", alpha=0.15)

    # Annotate the morning + evening commute peaks on the first weekday in the window.
    wk = w[~w["weekend"]]
    if len(wk):
        d0 = pd.Timestamp(wk["date"].iloc[0])
        for hr, txt, dy in ((8, "AM commute peak", 60), (18, "PM commute peak", 60)):
            row = w[w["dt"] == d0 + pd.Timedelta(hours=hr)]
            if len(row):
                yv = float(row["actual"].iloc[0])
                ax.annotate(txt, (d0 + pd.Timedelta(hours=hr), yv),
                            xytext=(0, dy), textcoords="offset points", ha="center",
                            fontsize=8, color="dimgray",
                            arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8))

    ax.set_title(f"Best-tracking week: {start:%b %d} – {(end - pd.Timedelta(days=1)):%b %d, %Y}  "
                 f"(of the known Nov 15–30 window)\n"
                 f"RMSE {rmse:.1f}, R² {r2:.3f}, {within:.0f}% of hours within 20% — clean "
                 f"weekday commute peaks, lower weekend hump", fontsize=13)
    ax.set_xlabel("Day  (light bands = rush hours, gray = weekend)")
    ax.set_ylabel("Rentals / hour")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%b %d"))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_best_week_zoom.png", dpi=120)
    plt.close()
    return start, end, rmse, r2


def main():
    os.makedirs(OUT, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load()

    fig_timeline(df)
    fig_daily(df)
    fig_profiles(df)
    fig_heatmap(df)
    wk_start, wk_end, wk_rmse, wk_r2 = fig_best_week(df)
    print(f"best week: {wk_start:%b %d} – {(wk_end - pd.Timedelta(days=1)):%b %d}  "
          f"RMSE {wk_rmse:.1f}, R² {wk_r2:.3f}")

    # Headline numbers for the writeup.
    gt = df[df["actual"].notna()]
    print(f"span: {df['dt'].min():%Y-%m-%d} → {df['dt'].max():%Y-%m-%d}  ({df['date'].nunique()} days)")
    print(f"ground-truth window RMSE {root_mean_squared_error(gt['actual'], gt['v3']):.1f}, "
          f"R² {r2_score(gt['actual'], gt['v3']):.3f}")
    by = df.groupby("day_type")["v3"].mean()
    print("mean predicted/hr by day type:", {k: round(v, 1) for k, v in by.items()})
    xmas = df[df["date"].between("2023-12-23", "2023-12-26")]["v3"].mean()
    base = df[df["day_type"] == "Weekday"]["v3"].mean()
    print(f"Christmas-window mean {xmas:.1f} vs weekday baseline {base:.1f} "
          f"({100*(xmas/base-1):+.0f}%)")
    print(f"plots -> {OUT}/ (01_timeline_hourly, 02_daily_totals, 03_hour_profiles, 04_heatmap_day_hour)")


if __name__ == "__main__":
    main()
