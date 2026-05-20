"""
build_dashboard.py
==================
Generates `housing_dashboard.html` from `data/training_data/housing.csv`.

Pipeline
--------
1. Load CSV, parse `date`, derive features (age, was_renovated, price_per_sqft, log_price).
2. Compute per-zipcode aggregates + convex-hull polygons (lat/long) for the choropleth.
3. Compute temporal aggregates (monthly + weekday + quarter).
4. Compute feature distributions (binned), scatter samples, correlation matrix, outlier flags.
5. Embed everything as JSON into `dashboard_template.html` -> `housing_dashboard.html`.

Run
---
    python3 build_dashboard.py
        --csv  ../data/training_data/housing.csv
        --out  housing_dashboard.html

Extension points
----------------
* `DERIVED_FEATURES`  -> add engineered columns (they auto-appear in the feature pickers).
* `aggregate_zipcode` -> add new per-zipcode metrics; expose in the map metric dropdown
  by adding the key to `MAP_METRICS` in `dashboard_template.html`.
* `bin_feature`       -> change histogram bin strategy.
* `SCATTER_SAMPLE_N`  -> trade interactivity for fidelity in the scatter view.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCATTER_SAMPLE_N = 4000          # rows kept for scatter views (perf vs fidelity)
HISTOGRAM_BINS = 40
OUTLIER_IQR_K = 3.0              # k * IQR for outlier flag
DEFAULT_CSV = Path(__file__).parent.parent / "data" / "training_data" / "housing.csv"
DEFAULT_OUT = Path(__file__).parent / "housing_dashboard.html"
TEMPLATE = Path(__file__).parent / "dashboard_template.html"

# Feature dictionary: db_name -> (display_name, dtype, group)
# `group` drives which chart picker each feature appears in.
FEATURES = {
    "price":         ("Sale price (USD)",                  "money",   "target"),
    "bedrooms":      ("Bedrooms",                          "int",     "structure"),
    "bathrooms":     ("Bathrooms",                         "float",   "structure"),
    "sqft_living":   ("Living area (sq ft)",               "int",     "structure"),
    "sqft_lot":      ("Lot size (sq ft)",                  "int",     "structure"),
    "floors":        ("Floors (number of stories)",        "float",   "structure"),
    "waterfront":    ("Waterfront (0/1)",                  "bool",    "amenity"),
    "view":          ("View quality (0–4)",                "ord",     "amenity"),
    "condition":     ("Condition (1–5)",                   "ord",     "quality"),
    "grade":         ("Construction grade (1–13)",         "ord",     "quality"),
    "sqft_above":    ("Above-ground area (sq ft)",         "int",     "structure"),
    "sqft_basement": ("Basement area (sq ft)",             "int",     "structure"),
    "yr_built":      ("Year built",                        "int",     "time"),
    "yr_renovated":  ("Year renovated (0 = never)",        "int",     "time"),
    "zipcode":       ("ZIP code",                          "cat",     "geo"),
    "lat":           ("Latitude",                          "float",   "geo"),
    "long":          ("Longitude",                         "float",   "geo"),
    "sqft_living15": ("Neighbors' living area (sq ft)",    "int",     "context"),
    "sqft_lot15":    ("Neighbors' lot size (sq ft)",       "int",     "context"),
    # Derived
    "age":           ("Age at sale (years)",               "int",     "derived"),
    "was_renovated": ("Renovated (0/1)",                   "bool",    "derived"),
    "price_per_sqft":("Price per sq ft (USD)",             "money",   "derived"),
    "log_price":     ("log10(price)",                      "float",   "derived"),
    "basement":      ("Has basement (0/1)",                "bool",    "derived"),
    "sale_month":    ("Sale month (1–12)",                 "int",     "derived"),
    "sale_year":     ("Sale year",                         "int",     "derived"),
}

NUMERIC_KEYS = [k for k, v in FEATURES.items() if v[1] in ("int", "float", "money", "ord", "bool")]


# -----------------------------------------------------------------------------
# Loaders / feature engineering
# -----------------------------------------------------------------------------
def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["sale_weekday"] = df["date"].dt.day_name()
    df["age"] = df["sale_year"] - df["yr_built"]
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["basement"] = (df["sqft_basement"] > 0).astype(int)
    df["price_per_sqft"] = df["price"] / df["sqft_living"].clip(lower=1)
    df["log_price"] = np.log10(df["price"].clip(lower=1))
    # Repair the well-known 33-bedroom data-entry error (single row)
    df.loc[df["bedrooms"] > 11, "bedrooms"] = df["bedrooms"].median()
    return df


# -----------------------------------------------------------------------------
# Aggregations
# -----------------------------------------------------------------------------
def load_external_geojson(path: Path) -> dict[int, list]:
    """If `kc_zipcodes.geojson` exists, return {zipcode -> outer ring as [[lon,lat], ...]}.

    The OpenDataDE WA file uses `ZCTA5CE10` for the zip code; we tolerate a few
    common alternates.
    """
    if not path.exists():
        return {}
    gj = json.loads(path.read_text())
    keys = ("ZCTA5CE10", "ZCTA5CE20", "ZIP", "ZIPCODE", "zip", "zipcode")
    out = {}
    for f in gj.get("features", []):
        props = f.get("properties", {})
        zc = None
        for k in keys:
            if k in props:
                try:
                    zc = int(props[k]);  break
                except (TypeError, ValueError):
                    continue
        if zc is None:
            continue
        geom = f.get("geometry", {})
        if geom.get("type") == "Polygon":
            ring = geom["coordinates"][0]
        elif geom.get("type") == "MultiPolygon":
            # take the largest outer ring
            ring = max((p[0] for p in geom["coordinates"]), key=len)
        else:
            continue
        out[zc] = [[float(x), float(y)] for x, y in ring]
    return out


def aggregate_zipcode(df: pd.DataFrame) -> list[dict]:
    """Per-zipcode KPIs + polygon (real GeoJSON if available, else convex hull)."""
    external = load_external_geojson(Path(__file__).parent / "kc_zipcodes.geojson")
    out = []
    for zc, g in df.groupby("zipcode"):
        if int(zc) in external:
            ring = external[int(zc)]
        else:
            pts = g[["long", "lat"]].to_numpy()
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    ring = pts[hull.vertices].tolist()
                    ring.append(ring[0])  # close
                except Exception:
                    ring = None
            else:
                ring = None

        out.append({
            "zipcode": int(zc),
            "n": int(len(g)),
            "median_price": float(g["price"].median()),
            "mean_price": float(g["price"].mean()),
            "median_ppsf": float(g["price_per_sqft"].median()),
            "median_sqft": float(g["sqft_living"].median()),
            "median_grade": float(g["grade"].median()),
            "median_age": float(g["age"].median()),
            "pct_waterfront": float(g["waterfront"].mean() * 100),
            "pct_renovated": float(g["was_renovated"].mean() * 100),
            "center_lat": float(g["lat"].mean()),
            "center_long": float(g["long"].mean()),
            "polygon": ring,
        })
    return out


def aggregate_temporal(df: pd.DataFrame) -> dict:
    monthly = (df.groupby(df["date"].dt.to_period("M"))
                 .agg(n=("price", "size"),
                      median_price=("price", "median"),
                      mean_ppsf=("price_per_sqft", "mean"))
                 .reset_index())
    monthly["month"] = monthly["date"].astype(str)
    monthly = monthly.drop(columns=["date"]).to_dict(orient="list")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
    weekday = (df.groupby("sale_weekday")
                 .agg(n=("price", "size"), median_price=("price", "median"))
                 .reindex(weekday_order)
                 .reset_index()
                 .to_dict(orient="list"))

    yr_built_cohort = (df.assign(decade=(df["yr_built"] // 10) * 10)
                         .groupby("decade")
                         .agg(n=("price", "size"),
                              median_price=("price", "median"),
                              median_ppsf=("price_per_sqft", "median"))
                         .reset_index()
                         .to_dict(orient="list"))

    return {"monthly": monthly, "weekday": weekday, "yr_built_decade": yr_built_cohort}


def correlation_matrix(df: pd.DataFrame) -> dict:
    keys = ["price", "log_price", "price_per_sqft", "sqft_living", "sqft_lot",
            "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15",
            "bedrooms", "bathrooms", "floors", "grade", "condition", "view",
            "waterfront", "age", "was_renovated", "lat", "long"]
    corr = df[keys].corr().round(3)
    return {"keys": keys, "matrix": corr.values.tolist()}


def bin_feature(values: np.ndarray, bins: int = HISTOGRAM_BINS) -> dict:
    """Quantile-clipped histogram (drops top/bottom 0.5% to keep bars meaningful)."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return {"bin_edges": [], "counts": []}
    lo, hi = np.quantile(v, [0.005, 0.995])
    if lo == hi:
        hi = lo + 1
    v = v[(v >= lo) & (v <= hi)]
    counts, edges = np.histogram(v, bins=bins)
    return {"bin_edges": [float(x) for x in edges],
            "counts":    [int(x) for x in counts]}


def distributions(df: pd.DataFrame) -> dict:
    out = {}
    for k in NUMERIC_KEYS:
        if k not in df.columns:
            continue
        out[k] = bin_feature(df[k].to_numpy(dtype=float))
        out[k]["skew"] = float(pd.Series(df[k]).skew())
    return out


def scatter_sample(df: pd.DataFrame) -> list[dict]:
    n = min(SCATTER_SAMPLE_N, len(df))
    sample = df.sample(n=n, random_state=42)
    cols = ["sqft_living", "sqft_lot", "bedrooms", "bathrooms", "grade",
            "condition", "view", "age", "lat", "long", "price",
            "log_price", "price_per_sqft", "waterfront", "zipcode"]
    return sample[cols].round(4).to_dict(orient="records")


def outlier_summary(df: pd.DataFrame) -> dict:
    out = {}
    for k in ["price", "sqft_living", "sqft_lot", "bedrooms", "bathrooms",
              "price_per_sqft"]:
        s = df[k]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - OUTLIER_IQR_K * iqr, q3 + OUTLIER_IQR_K * iqr
        flagged = ((s < lo) | (s > hi)).sum()
        out[k] = {
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower_fence": float(lo), "upper_fence": float(hi),
            "n_flagged": int(flagged), "pct_flagged": float(flagged / len(s) * 100),
            "min": float(s.min()), "max": float(s.max()),
            "median": float(s.median()), "mean": float(s.mean()),
        }
    return out


def kpis(df: pd.DataFrame) -> dict:
    return {
        "n_listings": int(len(df)),
        "n_zipcodes": int(df["zipcode"].nunique()),
        "median_price": float(df["price"].median()),
        "mean_price": float(df["price"].mean()),
        "median_ppsf": float(df["price_per_sqft"].median()),
        "pct_waterfront": float(df["waterfront"].mean() * 100),
        "pct_renovated": float(df["was_renovated"].mean() * 100),
        "date_min": df["date"].min().strftime("%Y-%m-%d"),
        "date_max": df["date"].max().strftime("%Y-%m-%d"),
        "median_grade": float(df["grade"].median()),
        "median_age": float(df["age"].median()),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    args = ap.parse_args()

    df = load(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")

    payload = {
        "meta": {
            "source": str(args.csv.name),
            "n_rows": int(len(df)),
            "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        },
        "features":      {k: {"label": v[0], "dtype": v[1], "group": v[2]}
                          for k, v in FEATURES.items()},
        "kpis":          kpis(df),
        "zipcodes":      aggregate_zipcode(df),
        "temporal":      aggregate_temporal(df),
        "correlations":  correlation_matrix(df),
        "distributions": distributions(df),
        "outliers":      outlier_summary(df),
        "scatter":       scatter_sample(df),
    }

    template = args.template.read_text()
    html = template.replace(
        "/* __DATA__ */",
        f"const DATA = {json.dumps(payload, separators=(',', ':'))};"
    )
    args.out.write_text(html)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"Wrote {args.out}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
