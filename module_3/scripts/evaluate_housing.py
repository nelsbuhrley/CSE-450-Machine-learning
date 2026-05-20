#!/usr/bin/env python3
"""
Housing Prediction Evaluator for CSE-450 Module 3.

Evaluates housing price prediction CSVs against the mini holdout answer key.

Usage as CLI:
    python evaluate_housing.py                          # evaluate all CSVs in default dir
    python evaluate_housing.py path/to/predictions/     # evaluate all CSVs in a directory
    python evaluate_housing.py my_predictions.csv       # evaluate a single file
    python evaluate_housing.py --plot                   # show scatter plots

Usage as library:
    from evaluate_housing import load_answers, evaluate_predictions, evaluate_directory

    answers = load_answers()
    results = evaluate_predictions(answers, "my_predictions.csv")
    print(results["R2"], results["RMSE"])
"""

from pathlib import Path
import argparse
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

# ── Paths ────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _SCRIPT_DIR.parent

DEFAULT_ANSWERS_PATH = _MODULE_DIR / "data" / "verification_data" / "housing_holdout_test_mini_answers.csv"
DEFAULT_PREDICTIONS_DIR = _MODULE_DIR / "predictions" / "mini_holdout"


# ── Core Functions ───────────────────────────────────────────────────────────

def load_answers(path: Path | str | None = None) -> pd.DataFrame:
    """Load the answer key CSV. Returns a DataFrame with a single 'price' column."""
    path = Path(path) if path else DEFAULT_ANSWERS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Answer key not found: {path}")
    df = pd.read_csv(path)
    df.columns = ["price"]
    return df


def evaluate_predictions(
    answers: pd.DataFrame,
    predictions_path: Path | str,
    label: str | None = None,
) -> dict:
    """
    Evaluate a single predictions CSV against the answer key.

    Parameters
    ----------
    answers : DataFrame
        The answer key (from load_answers()).
    predictions_path : Path or str
        Path to a CSV with a single column of predicted prices.
    label : str, optional
        Human-readable label for this prediction set.
        Defaults to the filename stem.

    Returns
    -------
    dict with keys:
        label, RMSE, MAE, MedianAE, R2,
        within_5_pct, within_10_pct, within_20_pct,
        detail  (DataFrame with per-row comparison)
    """
    predictions_path = Path(predictions_path)
    if label is None:
        label = predictions_path.stem

    preds = pd.read_csv(predictions_path)

    if preds.shape[0] != answers.shape[0]:
        raise ValueError(
            f"{label}: expected {answers.shape[0]} rows, got {preds.shape[0]}"
        )

    # Normalize to single column named 'price'
    preds = preds.iloc[:, :1]
    preds.columns = ["price"]

    # Metrics
    rmse = root_mean_squared_error(answers["price"], preds["price"])
    mae = mean_absolute_error(answers["price"], preds["price"])
    median_ae = median_absolute_error(answers["price"], preds["price"])
    r2 = r2_score(answers["price"], preds["price"])

    # Per-row detail
    detail = pd.DataFrame({
        "actual": answers["price"].values,
        "predicted": preds["price"].values,
    })
    detail["abs_error"] = (detail["actual"] - detail["predicted"]).abs()
    detail["abs_error_pct"] = detail["abs_error"] / detail["actual"]

    # Percent-within buckets
    n = len(detail)
    within = {}
    for pct in (5, 10, 20):
        within[pct] = (detail["abs_error_pct"] <= pct / 100).sum() / n * 100

    return {
        "label": label,
        "RMSE": rmse,
        "MAE": mae,
        "MedianAE": median_ae,
        "R2": r2,
        "within_5_pct": within[5],
        "within_10_pct": within[10],
        "within_20_pct": within[20],
        "detail": detail,
    }


def evaluate_directory(
    answers: pd.DataFrame,
    directory: Path | str | None = None,
) -> list[dict]:
    """
    Evaluate every CSV in a directory. Returns a list of result dicts.
    """
    directory = Path(directory) if directory else DEFAULT_PREDICTIONS_DIR
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return []

    results = []
    for csv_path in csv_files:
        try:
            result = evaluate_predictions(answers, csv_path)
            results.append(result)
        except Exception as e:
            print(f"  SKIP {csv_path.name}: {e}", file=sys.stderr)

    return results


def summary_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert a list of result dicts into a tidy summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Team/File": r["label"],
            "RMSE": round(r["RMSE"], 2),
            "MAE": round(r["MAE"], 2),
            "Median AE": round(r["MedianAE"], 2),
            "R²": round(r["R2"], 4),
            "Within 5%": f"{r['within_5_pct']:.1f}%",
            "Within 10%": f"{r['within_10_pct']:.1f}%",
            "Within 20%": f"{r['within_20_pct']:.1f}%",
        })
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


# ── Display ──────────────────────────────────────────────────────────────────

def print_results(results: list[dict]) -> None:
    """Print formatted results to stdout."""
    for r in results:
        print(f"\n{'─' * 30} {r['label'].upper()} {'─' * 30}")
        print(f"  RMSE:              {r['RMSE']:,.2f}")
        print(f"  Mean Abs Error:    {r['MAE']:,.2f}")
        print(f"  Median Abs Error:  {r['MedianAE']:,.2f}")
        print(f"  R²:               {r['R2']:.4f}")
        print(f"  Within  5%:       {r['within_5_pct']:.1f}%")
        print(f"  Within 10%:       {r['within_10_pct']:.1f}%")
        print(f"  Within 20%:       {r['within_20_pct']:.1f}%")

    if len(results) > 1:
        print(f"\n{'═' * 72}")
        print("SUMMARY (sorted by R²):\n")
        print(summary_dataframe(results).to_string(index=False))


def plot_results(results: list[dict]) -> None:
    """Show a scatter plot (actual vs predicted) for each result set."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(rc={"figure.figsize": (10, 8)})

    for r in results:
        detail = r["detail"]
        detail["bucket"] = np.where(
            detail["abs_error_pct"] >= 0.20, "above 20%", "below 20%"
        )
        palette = {"below 20%": "tab:blue", "above 20%": "tab:orange"}
        lim = (0, max(detail["actual"].max(), detail["predicted"].max()) * 1.05)

        ax = sns.scatterplot(
            data=detail, x="actual", y="predicted", hue="bucket", palette=palette,
        )
        ax.plot(lim, lim, color="red", linewidth=1, label="perfect")
        ax.set(xlim=lim, ylim=lim)
        ax.set_title(r["label"])
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate housing price predictions against the mini holdout answer key.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_PREDICTIONS_DIR),
        help="Path to a predictions CSV or a directory of CSVs (default: predictions/mini_holdout/)",
    )
    parser.add_argument(
        "--answers",
        default=str(DEFAULT_ANSWERS_PATH),
        help="Path to the answer key CSV (default: data/verification_data/housing_holdout_test_mini_answers.csv)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show scatter plots for each prediction set",
    )
    parser.add_argument(
        "--csv",
        metavar="OUTPUT",
        help="Save summary table to a CSV file",
    )

    args = parser.parse_args()
    answers = load_answers(args.answers)
    target = Path(args.target)

    if target.is_file():
        results = [evaluate_predictions(answers, target)]
    elif target.is_dir():
        results = evaluate_directory(answers, target)
    else:
        print(f"Error: {target} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    if not results:
        sys.exit(0)

    print_results(results)

    if args.csv:
        summary_dataframe(results).to_csv(args.csv, index=False)
        print(f"\nSummary saved to {args.csv}")

    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
