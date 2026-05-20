"""
import_geojson.py
=================
Downloads the OpenDataDE Washington State zipcode GeoJSON, filters it to the
zipcodes that appear in `housing.csv`, and writes a trimmed copy to
`dashboards/kc_zipcodes.geojson` (where `build_dashboard.py` looks for it).

Why filter
----------
The full WA file is ~3 MB and contains ~700 zipcodes. Our dataset uses only
~70 King County zipcodes, so the trimmed file is ~10x smaller and keeps the
final dashboard HTML lean.

Run
---
    python3 import_geojson.py

Options
-------
    --csv      Path to housing.csv (default: ../data/training_data/housing.csv)
    --out      Output GeoJSON (default: ./kc_zipcodes.geojson)
    --url      Source URL (default: OpenDataDE WA file)
    --keep-all Skip filtering — store the full state file.

If the download fails (firewall, offline), the script prints exactly which
domain is unreachable and exits cleanly. `build_dashboard.py` will then fall
back to its convex-hull approximation without complaint.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_URL = (
    "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/"
    "master/wa_washington_zip_codes_geo.min.json"
)
HERE = Path(__file__).parent


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "kc-housing-dashboard"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv",  type=Path, default=HERE.parent / "data" / "training_data" / "housing.csv")
    ap.add_argument("--out",  type=Path, default=HERE / "kc_zipcodes.geojson")
    ap.add_argument("--url",  type=str,  default=DEFAULT_URL)
    ap.add_argument("--keep-all", action="store_true")
    args = ap.parse_args()

    print(f"Fetching  {args.url}")
    try:
        gj = fetch(args.url)
    except Exception as e:
        print(f"  ERROR — {type(e).__name__}: {e}", file=sys.stderr)
        print("  build_dashboard.py will fall back to convex-hull polygons.", file=sys.stderr)
        return 1
    print(f"  got {len(gj.get('features', []))} features")

    # Filter
    if args.keep_all:
        out = gj
    else:
        zips_in_data = set(pd.read_csv(args.csv, usecols=["zipcode"])["zipcode"].astype(int).unique())
        print(f"  filtering to {len(zips_in_data)} zipcodes from {args.csv.name}")
        keys = ("ZCTA5CE10", "ZCTA5CE20", "ZIP", "ZIPCODE")
        kept = []
        for f in gj["features"]:
            props = f.get("properties", {})
            for k in keys:
                if k in props:
                    try:
                        if int(props[k]) in zips_in_data:
                            kept.append(f)
                    except (ValueError, TypeError):
                        pass
                    break
        out = {"type": "FeatureCollection", "features": kept}
        print(f"  kept {len(kept)} features")

    args.out.write_text(json.dumps(out, separators=(",", ":")))
    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote     {args.out}  ({size_kb:.0f} KB)")
    print("Now re-run: python3 build_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
