"""V3 feature engineering: holiday-PROXIMITY features on top of the V2 pipeline.

The V2 model only knows the single day a holiday falls on (the raw `holiday` flag).
But demand is depressed across the whole *holiday neighbourhood* — the day before
(people travel), the bridge day after (e.g. Black Friday, the Fri after Thanksgiving),
and the surrounding week. Those days are still tagged workingday=1, so the V2 model
predicts full commute peaks on them and over-predicts (the Nov 21-24 over-prediction
in the mini holdout is exactly this).

We add features off a canonical "observed weekday day-off" calendar (US-DC federal
schedule), which matches the data's own `holiday` flags ~99% but, crucially, is defined
for ANY date — so adjacency is correct even at the holdout edges, where the nearest
holiday (Veterans Day, Nov 10 2023) sits outside the Nov 15-30 window.
"""

import functools

import numpy as np
import pandas as pd

import module4 as m

NEAR_DAYS = 3  # a working day within this many days of a day-off has suppressed commute


@functools.lru_cache(maxsize=8)
def canonical_dayoffs(year_lo: int, year_hi: int) -> np.ndarray:
    """Sorted observed weekday day-offs (US-DC schedule) as datetime64[D].

    Weekday-only because the data's `holiday` flag marks the *day off work*, not
    holidays that land on a weekend. Matches the training flags within a few days
    over 13 years, and extends cleanly past the data for edge-correct adjacency.
    """
    import holidays

    cal = holidays.US(years=range(year_lo, year_hi + 1), subdiv="DC", observed=True)
    days = sorted(d for d in cal.keys() if d.weekday() < 5)
    return np.array([np.datetime64(d) for d in days], dtype="datetime64[D]")


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday-proximity features. Requires `dteday` (datetime) and `dow` columns."""
    from datetime import timedelta

    dates = df["dteday"].dt.normalize()
    d64 = dates.values.astype("datetime64[D]")
    yrs = (int(dates.dt.year.min()) - 1, int(dates.dt.year.max()) + 1)
    offs = canonical_dayoffs(*yrs)
    off_set = set(offs.tolist())  # datetime.date keys

    pyd = d64.tolist()  # datetime.date — compare in the same type as off_set
    one = timedelta(days=1)
    n = len(pyd)
    weekday = (df["dow"] < 5).to_numpy()
    is_off = np.fromiter((x in off_set for x in pyd), bool, n)
    prev_off = np.fromiter(((x - one) in off_set for x in pyd), bool, n)
    next_off = np.fromiter(((x + one) in off_set for x in pyd), bool, n)
    prev_wknd = np.fromiter(((x - one).weekday() >= 5 for x in pyd), bool, n)
    next_wknd = np.fromiter(((x + one).weekday() >= 5 for x in pyd), bool, n)

    # Distance (days) to the nearest day-off, via the sorted calendar.
    idx = np.searchsorted(offs, d64)
    left = np.clip(idx - 1, 0, len(offs) - 1)
    right = np.clip(idx, 0, len(offs) - 1)
    dist_l = np.abs((d64 - offs[left]).astype(int))
    dist_r = np.abs((offs[right] - d64).astype(int))
    days_to = np.minimum(dist_l, dist_r)

    df["is_dayoff"] = is_off.astype(int)
    df["day_after_holiday"] = (prev_off & weekday & ~is_off).astype(int)
    df["day_before_holiday"] = (next_off & weekday & ~is_off).astype(int)
    # Bridge day: a workday wedged between a day-off and a weekend (classic Black Friday).
    df["bridge_day"] = (
        weekday & ~is_off & ((prev_off & next_wknd) | (next_off & prev_wknd))
    ).astype(int)
    df["days_to_holiday"] = np.minimum(days_to, 14)
    df["holiday_proximity"] = 1.0 - np.minimum(days_to, 7) / 7.0
    # Working day in a holiday's neighbourhood — its commute peak is suppressed.
    suppressed = ((days_to <= NEAR_DAYS) & (df["workingday"].to_numpy() == 1)).astype(int)
    df["holiday_week"] = suppressed
    # Give the model a dedicated hour channel to dampen the rush peak on these days.
    df["hour_sin_holweek"] = df["hour_sin"] * suppressed
    df["hour_cos_holweek"] = df["hour_cos"] * suppressed
    return df


def engineer_v3(url: str) -> pd.DataFrame:
    return add_holiday_features(m.load_and_engineer_features(url))


def feature_cols_v3(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("dteday", "hr", "dow", "casual", "registered")]
