# Module 3 – Housing Prediction Evaluator

Evaluates housing price prediction CSVs against the mini holdout answer key.

## Setup

Requires Python 3.10+ and the following packages (likely already installed):

```
pandas numpy scikit-learn matplotlib seaborn
```

## Quick Start (Command Line)

```bash
# From the module_3/scripts/ directory:

# Evaluate all CSVs in the default predictions folder (predictions/mini_holdout/)
python evaluate_housing.py

# Evaluate a single file
python evaluate_housing.py ../predictions/mini_holdout/my_predictions.csv

# Evaluate a different directory
python evaluate_housing.py path/to/some/folder/

# Show scatter plots
python evaluate_housing.py --plot

# Save summary to CSV
python evaluate_housing.py --csv results.csv

# Use a different answer key
python evaluate_housing.py --answers path/to/answers.csv
```

## How to Add Your Predictions

1. Generate a CSV with a single column called `price` containing your predicted prices — one per row, 81 total (matching the holdout set).
2. Drop the CSV into `module_3/predictions/mini_holdout/`.
3. Run `python evaluate_housing.py`.

Example CSV format:
```
price
350000
425000
280000
...
```

## Using in Your Own Scripts

```python
from evaluate_housing import load_answers, evaluate_predictions, evaluate_directory, summary_dataframe

# Load the answer key (uses the local CSV by default)
answers = load_answers()

# Evaluate a single predictions file
result = evaluate_predictions(answers, "my_predictions.csv")
print(result["R2"])       # 0.78
print(result["RMSE"])     # 133739.94
print(result["within_10_pct"])  # 44.4 (percent)

# Access per-row detail DataFrame
detail = result["detail"]  # columns: actual, predicted, abs_error, abs_error_pct

# Evaluate all CSVs in a directory
results = evaluate_directory(answers, "../predictions/mini_holdout")

# Get a tidy summary DataFrame
df = summary_dataframe(results)
print(df)
```

### Plotting from a Script

```python
from evaluate_housing import load_answers, evaluate_predictions, plot_results

# Plot a single prediction file
answers = load_answers()
results = [evaluate_predictions(answers, "my_predictions.csv")]
plot_results(results)

# Or plot all predictions in a directory at once
from evaluate_housing import evaluate_directory

results = evaluate_directory(answers)
plot_results(results)
```

### Key Functions

| Function | Purpose |
|---|---|
| `load_answers(path=None)` | Load answer key CSV. Defaults to the local mini answers file. |
| `evaluate_predictions(answers, csv_path, label=None)` | Evaluate one CSV. Returns dict with RMSE, MAE, MedianAE, R², within-percent metrics, and a detail DataFrame. |
| `evaluate_directory(answers, directory=None)` | Evaluate every CSV in a directory. Returns list of result dicts. |
| `summary_dataframe(results)` | Convert result dicts to a sorted summary DataFrame. |
| `print_results(results)` | Pretty-print results to terminal. |
| `plot_results(results)` | Show actual-vs-predicted scatter plots. |

### Result Dict Keys

```python
{
    "label":          str,    # filename stem or custom label
    "RMSE":           float,
    "MAE":            float,
    "MedianAE":       float,
    "R2":             float,
    "within_5_pct":   float,  # percent of predictions within 5% of actual
    "within_10_pct":  float,
    "within_20_pct":  float,
    "detail":         pd.DataFrame,  # per-row: actual, predicted, abs_error, abs_error_pct
}
```

## Metrics Explained

| Metric | What It Tells You |
|---|---|
| **RMSE** | Root Mean Squared Error — penalizes large errors heavily. Lower is better. |
| **MAE** | Mean Absolute Error — average dollar amount off. Lower is better. |
| **Median AE** | Median Absolute Error — typical error (robust to outliers). Lower is better. |
| **R²** | How much variance your model explains. 1.0 = perfect, 0.0 = baseline. |
| **Within X%** | Percentage of predictions within X% of the actual price. Higher is better. |
