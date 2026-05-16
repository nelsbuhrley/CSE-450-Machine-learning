"""
Value-per-call efficiency metric.

Compares each model's value-per-call against the call-everyone baseline.
Called by main_vis.py — not intended to run standalone.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CALL_COST = -11 * 0.5
TERM_DEPOSIT_BENEFIT = 4960 * 0.75 * 0.012

HERE = Path(__file__).parent
PREDICTIONS_DIR = HERE.parent / "predictions" / "mini-holdout"
ANSWERS_FILE = HERE.parent / "verification_data" / "bank_holdout_test_mini_answers.csv"


def value_of_calls(fp: int, tp: int) -> float:
    return (fp + tp) * CALL_COST + tp * TERM_DEPOSIT_BENEFIT


def value_per_call(fp: int, tp: int) -> float:
    total = fp + tp
    return value_of_calls(fp, tp) / total if total > 0 else 0.0


def extract_name(path: Path) -> str:
    stem = path.stem
    inner = stem.removeprefix("NorthWindModule2_")
    inner = inner.rsplit("_mini-holdout-predictions", 1)[0]
    parts = inner.split("_", 2)
    if len(parts) == 3:
        n, person, model_type = parts
        return f"{n} {person} ({model_type})"
    return inner


def run_analysis() -> dict:
    answers = pd.read_csv(ANSWERS_FILE).iloc[:, 0]
    expected_len = len(answers)
    tp_all = int(answers.sum())
    fp_all = int((answers == 0).sum())
    baseline_vpc = value_per_call(fp_all, tp_all)

    results = []
    for pred_file in sorted(PREDICTIONS_DIR.glob("*.csv")):
        name = extract_name(pred_file)
        preds = pd.read_csv(pred_file).iloc[:, 0]
        if len(preds) != expected_len:
            continue
        cm = confusion_matrix(preds, answers)
        fp = int(cm[1][0]); tp = int(cm[1][1])
        total_calls = tp + fp
        vpc = value_per_call(fp, tp)
        results.append({
            "name": name, "total_calls": total_calls,
            "tp": tp, "fp": fp, "vpc": round(vpc, 2),
            "lift": round(vpc - baseline_vpc, 2),
        })

    results.sort(key=lambda r: r["vpc"], reverse=True)
    return {"baseline_vpc": round(baseline_vpc, 2), "baseline_tp": tp_all,
            "baseline_fp": fp_all, "models": results}


def make_figures(data: dict) -> dict[str, plt.Figure]:
    baseline_vpc = data["baseline_vpc"]
    names = [r["name"] for r in data["models"]]
    vpcs = [r["vpc"] for r in data["models"]]
    colors = ["#2a9d8f" if v > baseline_vpc else "#DD8452" for v in vpcs]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    ax.bar(names, vpcs, color=colors, edgecolor="white")
    ax.axhline(baseline_vpc, color="#e76f51", linestyle="--", linewidth=1.5,
               label=f"Baseline / call everyone (${baseline_vpc:.2f})")
    ax.set_title("Value per Call vs Call-Everyone Baseline", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value per Call ($)")
    ax.set_xlabel("Model")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return {"value_per_call": fig}


def make_markdown(data: dict) -> str:
    lines = []
    lines.append("## Value per Call Analysis\n")
    lines.append(f"Baseline (call everyone): **${data['baseline_vpc']:.2f}/call** "
                 f"({data['baseline_tp']} TP, {data['baseline_fp']} FP)\n")
    lines.append("![Value per Call](value_per_call.png)\n")
    lines.append("| Model | Calls | TP | Value/Call | Lift over Baseline |")
    lines.append("|-------|------:|---:|----------:|-------------------:|")
    for r in data["models"]:
        lines.append(f"| {r['name']} | {r['total_calls']} | {r['tp']} | ${r['vpc']:.2f} | +${r['lift']:.2f} |")
    lines.append("")
    return "\n".join(lines)
