"""
Grade mini-holdout predictions against the answer key.

Business model: bank telemarketing campaign
  - Calling someone costs employee time (wage * 0.5h)
  - Correct call (true positive) earns interest on term deposit
  - Incorrect call (false positive) just costs time
  - Missed customer (false negative) is lost opportunity but no cost

Run from repo root or this directory — paths are relative to this file.
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- Scoring thresholds ---
BLUE_THRESHOLD = 650   # full credit
ORANGE_THRESHOLD = 300  # minimum passing

# --- Paths ---
HERE = Path(__file__).parent
PREDICTIONS_DIR = HERE.parent / "predictions" / "mini-holdout"
ANSWERS_FILE = HERE.parent / "verification_data" / "bank_holdout_test_mini_answers.csv"


def value_of_calls(false_positives: int, true_positives: int) -> float:
    """
    Estimated dollar value of the model's call decisions.

    Assumptions (from dataset time period):
      - 0.5h per call, $11/h teller wage
      - Average savings: $4,960
      - 75% of savings placed in term deposit
      - Net interest margin: 1.2%
    """
    call_cost = -11 * 0.5
    term_deposit_benefit = 4960 * 0.75 * 0.012
    return (false_positives + true_positives) * call_cost + true_positives * term_deposit_benefit


def estimated_grade(call_value: float) -> float:
    """Linear scale from ORANGE_THRESHOLD (0%) to BLUE_THRESHOLD (100%)."""
    if call_value >= BLUE_THRESHOLD:
        return 100.0
    if call_value <= ORANGE_THRESHOLD:
        return 0.0
    return (call_value - ORANGE_THRESHOLD) / (BLUE_THRESHOLD - ORANGE_THRESHOLD) * 100


def load_predictions(path: Path) -> pd.Series:
    """Read first numeric column regardless of header name."""
    df = pd.read_csv(path)
    return df.iloc[:, 0]


def extract_name(path: Path) -> str:
    """NorthWindModule2_{n}_{person}_{type}_mini-holdout-predictions.csv -> '{n} {person} ({type})'"""
    stem = path.stem  # NorthWindModule2_1_caleb_rf_mini-holdout-predictions
    # strip the fixed prefix and the trailing split+label
    inner = stem.removeprefix("NorthWindModule2_")
    inner = inner.rsplit("_mini-holdout-predictions", 1)[0]
    # inner = "1_caleb_rf" or "3_nels_stack_rf_knn"
    parts = inner.split("_", 2)  # [n, person, type]
    if len(parts) == 3:
        n, person, model_type = parts
        return f"{n} {person} ({model_type})"
    return inner


def grade_all() -> pd.DataFrame:
    answers = pd.read_csv(ANSWERS_FILE).iloc[:, 0]
    expected_len = len(answers)

    results = []
    for pred_file in sorted(PREDICTIONS_DIR.glob("*.csv")):
        name = extract_name(pred_file)
        preds = load_predictions(pred_file)

        if len(preds) != expected_len:
            print(f"  SKIP {name}: {len(preds)} rows (expected {expected_len})")
            continue

        # confusion_matrix(predictions, answers) gives us:
        #   cm[1][0] = called, didn't subscribe (false positive / wasted call)
        #   cm[1][1] = called, did subscribe   (true positive / good call)
        cm = confusion_matrix(preds, answers)
        false_pos = int(cm[1][0])
        true_pos = int(cm[1][1])
        call_val = value_of_calls(false_pos, true_pos)

        results.append({
            "Name": name,
            "Correct Calls (TP)": true_pos,
            "Wasted Calls (FP)": false_pos,
            "Value of Calls ($)": round(call_val, 2),
            "Estimated Grade (%)": round(estimated_grade(call_val), 1),
        })
        print(f"  Loaded: {name}")

    df = pd.DataFrame(results).set_index("Name")
    return df.sort_values("Value of Calls ($)", ascending=False)


def plot_results(df: pd.DataFrame):
    values = df["Value of Calls ($)"]
    colors = [
        "tab:blue" if v >= BLUE_THRESHOLD
        else "tab:red" if v < 0
        else "tab:orange" if v < ORANGE_THRESHOLD
        else "tab:grey"
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.5), 5))
    bars = ax.bar(df.index, values, color=colors)
    ax.axhline(BLUE_THRESHOLD, color="tab:blue", linestyle="--", linewidth=1, label=f"Blue ({BLUE_THRESHOLD})")
    ax.axhline(ORANGE_THRESHOLD, color="tab:orange", linestyle="--", linewidth=1, label=f"Orange ({ORANGE_THRESHOLD})")
    ax.set_title("Mini-Holdout: Estimated Value of Calls by Model")
    ax.set_ylabel("Value of Calls ($)")
    ax.set_xlabel("Model")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"Answers: {ANSWERS_FILE}")
    print(f"Predictions: {PREDICTIONS_DIR}\n")

    results_df = grade_all()

    print("\n--- Results ---")
    print(results_df.to_string())

    plot_results(results_df)
