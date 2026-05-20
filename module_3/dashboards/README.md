# King County Housing — EDA Dashboard

Self-contained HTML dashboard for the King County, WA housing dataset
(`data/training_data/housing.csv`). Built for the CSE 450 Module 3 ML
assignment — surface-area is EDA, not production analytics.

## Files

| File | Role |
|---|---|
| `housing_dashboard.html`    | **Generated** — open this in a browser. ~1 MB, no server. |
| `build_dashboard.py`        | Loads CSV, computes all aggregates, embeds JSON into the template. |
| `dashboard_template.html`   | HTML/CSS/JS shell. The string `/* __DATA__ */` is replaced with embedded data. |
| `import_geojson.py`         | Optional: pulls real WA zipcode polygons and trims to King County. |
| `kc_zipcodes.geojson`       | Optional input (created by `import_geojson.py`). If present, replaces convex hulls. |

## Regenerate

```bash
cd module_3/dashboards
python3 build_dashboard.py
# → writes housing_dashboard.html
```

## Add real zipcode polygons (optional, nicer-looking map)

Convex hulls are the default — zero external dependency. To swap in actual
zipcode shapes:

```bash
python3 import_geojson.py     # writes kc_zipcodes.geojson
python3 build_dashboard.py    # rebuilds with real polygons
```

If the fetch fails (firewall etc.), the build silently falls back to hulls.

## Data dictionary (db_name → display)

Numeric:
`price` (sale price USD) · `bedrooms` (Bedrooms) · `bathrooms` (Bathrooms) ·
`sqft_living` (Living area, sq ft) · `sqft_lot` (Lot size, sq ft) ·
`floors` (Stories) · `sqft_above` (Above-ground area, sq ft) ·
`sqft_basement` (Basement area, sq ft) · `yr_built` (Year built) ·
`yr_renovated` (Year renovated, 0 = never) · `lat` / `long` (Coords) ·
`sqft_living15` / `sqft_lot15` (Neighbors' median, 15 nearest).

Categorical / ordinal:
`waterfront` (0/1) · `view` (0–4 quality) · `condition` (1–5) ·
`grade` (King County construction grade, 1–13) · `zipcode` (ZIP code).

Derived (added in `build_dashboard.py`):
`age` (years at sale) · `was_renovated` (0/1) · `basement` (0/1) ·
`price_per_sqft` · `log_price` · `sale_year` / `sale_month` / `sale_weekday`.

## Dashboard surfaces

1. **KPIs** — listings, zipcodes, median/mean price, $/sqft, % waterfront, etc.
2. **Map** — choropleth + point overlay. Color by median price, $/sqft, count, grade, age, % waterfront. **Click a zipcode to filter every other chart.**
3. **Target distribution** — `price` and `log10(price)` histograms (motivates log target for linear models).
4. **Monthly volume & price** — bar + line, sales count vs median price.
5. **Year-built decade cohorts** — older stock isn't reliably cheaper.
6. **Day-of-week pattern** — closing-day bias.
7. **Scatter** — any feature × any feature, colored by a third. Default `sqft_living` × `price`.
8. **Correlation with price (bar)** — sorted by |r|.
9. **Full correlation matrix** — diverging red-blue heatmap to spot multicollinearity.
10. **Feature distributions** — small-multiples grid with skew flagged (|skew| > 1 → consider transform).
11. **Outlier & missingness summary** — IQR fences (k=3), counts, fractions.
12. **Zipcode detail table** — sortable, follows the active filter.

Toolbar filters (top-right): `year built ≥`, `bedrooms`, `grade ≥`,
`waterfront`, `price range`. Light/dark theme toggle persists in `localStorage`.

## Data assumptions / preprocessing

* 33-bedroom row (well-known data-entry error) → replaced with median.
* `yr_renovated = 0` is treated as "never renovated".
* `price_per_sqft` uses `sqft_living` (excludes lot).
* Histograms clip to 0.5–99.5 percentile per feature.
* Scatter view samples 4,000 rows (set via `SCATTER_SAMPLE_N`).
* Outlier flags: Tukey fences with k=3.

## Extending

| Want to… | Edit |
|---|---|
| Add a derived feature | `load()` in `build_dashboard.py`; add to `FEATURES` dict. |
| Add a new map metric | `aggregate_zipcode()` to compute it; add option to `#map-metric` `<select>` in `dashboard_template.html`. |
| Change histogram bin strategy | `bin_feature()` in `build_dashboard.py`. |
| Add a new chart | New `<canvas>` in `dashboard_template.html` + a `render*()` function called from `boot()`. Use `commonOpts()` for theme-aware styling. |
| Trade interactivity for fidelity | Raise `SCATTER_SAMPLE_N` (cost: file size). |
| Add a connector / live data | Replace the JSON embed with a `fetch()` in `boot()`. |

## Performance notes

* All data pre-aggregated in Python; the browser only filters and renders.
* Leaflet uses `preferCanvas: true` to keep point overlay smooth.
* No `localStorage` for data (only theme).
* Chart.js updates use `.destroy()`+rebuild on theme change for clean colour swaps.

## Known limitations

* Convex hulls aren't true zipcode boundaries (see `import_geojson.py` for the fix).
* No statistical model fit shown — that lives in `module_3/scripts/`.
* Toolbar filters update *most* charts, but feature-distribution mini-charts and the corr matrix use the full dataset (intentional — those are reference views).
