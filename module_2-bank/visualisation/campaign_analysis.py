"""
Campaign value analysis — three scenarios.

Returns structured results and matplotlib figures.
Called by main_vis.py — not intended to run standalone.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CALL_COST = -11 * 0.5
TERM_DEPOSIT_BENEFIT = 4960 * 0.75 * 0.012

HERE = Path(__file__).parent
ROOT = HERE.parent
TRAINING_FILE = ROOT / "training_data" / "bank.csv"
MINI_ANSWERS_FILE = ROOT / "verification_data" / "bank_holdout_test_mini_answers.csv"
MINI_PRED_DIR = ROOT / "predictions" / "mini-holdout"
LARGE_PRED_DIR = ROOT / "predictions" / "holdout"


def campaign_value(tp: float, fp: float) -> float:
    return (tp + fp) * CALL_COST + tp * TERM_DEPOSIT_BENEFIT


def extract_model_id(path: Path) -> str:
    stem = path.stem
    inner = stem.removeprefix("NorthWindModule2_")
    inner = inner.split("_holdout-predictions")[0].split("_mini-holdout-predictions")[0]
    parts = inner.split("_", 2)
    if len(parts) == 3:
        n, person, model_type = parts
        return f"{n} {person} ({model_type})"
    return inner


def model_key(path: Path) -> str:
    stem = path.stem
    inner = stem.removeprefix("NorthWindModule2_")
    return inner.split("_holdout-predictions")[0].split("_mini-holdout-predictions")[0]


def analyse_historical() -> dict:
    df = pd.read_csv(TRAINING_FILE)
    tp = int((df['y'] == 'yes').sum())
    fp = int((df['y'] == 'no').sum())
    total = tp + fp
    return {
        "total_called": total, "subscribed": tp,
        "conversion_rate": tp / total, "value": campaign_value(tp, fp),
    }


def analyse_mini_holdout() -> tuple[dict, list[dict]]:
    answers = pd.read_csv(MINI_ANSWERS_FILE, encoding='utf-8-sig').iloc[:, 0]
    total = len(answers)
    tp_all = int(answers.sum())
    fp_all = int((answers == 0).sum())
    baseline = {
        "label": "No model (call everyone)", "total_calls": total,
        "tp": tp_all, "fp": fp_all, "precision": tp_all / total,
        "value": campaign_value(tp_all, fp_all),
    }
    models = []
    for pred_file in sorted(MINI_PRED_DIR.glob("*.csv")):
        preds = pd.read_csv(pred_file).iloc[:, 0]
        if len(preds) != total:
            continue
        cm = confusion_matrix(preds, answers)
        fp = int(cm[1][0]); tp = int(cm[1][1])
        total_calls = tp + fp
        precision = tp / total_calls if total_calls > 0 else 0.0
        models.append({
            "label": extract_model_id(pred_file), "key": model_key(pred_file),
            "total_calls": total_calls, "tp": tp, "fp": fp,
            "precision": precision, "value": campaign_value(tp, fp),
        })
    return baseline, models


def project_large_holdout(mini_baseline: dict, mini_models: list[dict]) -> tuple[dict, list[dict]]:
    large_size = sum(1 for _ in open(ROOT / "test_data" / "bank_holdout_test.csv")) - 1
    conv_rate = mini_baseline["precision"]
    tp_est = large_size * conv_rate
    fp_est = large_size * (1 - conv_rate)
    baseline_proj = {
        "label": "No model (call everyone)", "total_calls": large_size,
        "tp_est": tp_est, "fp_est": fp_est,
        "value_projected": campaign_value(tp_est, fp_est),
        "note": f"conversion rate calibrated from mini holdout ({conv_rate:.1%})",
    }
    model_map = {m["key"]: m for m in mini_models}
    projections = []
    for pred_file in sorted(LARGE_PRED_DIR.glob("*.csv")):
        key = model_key(pred_file)
        if key not in model_map:
            continue
        preds = pd.read_csv(pred_file).iloc[:, 0]
        n_calls = int((preds == 1).sum())
        precision = model_map[key]["precision"]
        tp_est = n_calls * precision
        fp_est = n_calls * (1 - precision)
        projections.append({
            "label": extract_model_id(pred_file), "total_calls": n_calls,
            "tp_est": tp_est, "fp_est": fp_est,
            "precision_used": precision,
            "value_projected": campaign_value(tp_est, fp_est),
        })
    return baseline_proj, projections


def run_analysis() -> dict:
    """Return all data needed for markdown + plots."""
    hist = analyse_historical()
    mini_baseline, mini_models = analyse_mini_holdout()
    large_baseline, large_projections = project_large_holdout(mini_baseline, mini_models)
    return {
        "historical": hist,
        "mini_baseline": mini_baseline, "mini_models": mini_models,
        "large_baseline": large_baseline, "large_projections": large_projections,
    }


def make_comparison_chart(scenarios: list[tuple[str, float]], title: str):
    labels = [s[0] for s in scenarios]
    values = [s[1] for s in scenarios]
    colors = ["#aaaaaa" if "No model" in l else "#4C72B0" for l in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 1.8), 5))
    ax.bar(labels, values, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Campaign Value ($)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    return fig


def make_figures(data: dict) -> dict[str, plt.Figure]:
    mini_scenarios = [(data["mini_baseline"]["label"], data["mini_baseline"]["value"])]
    for m in data["mini_models"]:
        mini_scenarios.append((m["label"], m["value"]))

    large_scenarios = [(data["large_baseline"]["label"], data["large_baseline"]["value_projected"])]
    for p in data["large_projections"]:
        large_scenarios.append((p["label"], p["value_projected"]))

    return {
        "campaign_mini_holdout": make_comparison_chart(mini_scenarios, "Mini Holdout: No Model vs ML Models"),
        "campaign_large_holdout": make_comparison_chart(large_scenarios, "Large Holdout Projection: No Model vs ML Models"),
    }


def make_markdown(data: dict) -> str:
    lines = []
    lines.append("## Campaign Value Analysis\n")

    # Historical
    h = data["historical"]
    lines.append("### 1. Historical Campaign (training data)\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total called | {h['total_called']:,} |")
    lines.append(f"| Subscribed | {h['subscribed']:,} |")
    lines.append(f"| Conversion rate | {h['conversion_rate']:.1%} |")
    lines.append(f"| Campaign value | ${h['value']:,.2f} |")
    lines.append("")

    # Mini holdout
    lines.append("### 2. Mini Holdout (answers known)\n")
    lines.append("![Mini Holdout](campaign_mini_holdout.png)\n")
    mb = data["mini_baseline"]
    lines.append(f"| Model | Calls | TP | FP | Precision | Value |")
    lines.append(f"|-------|------:|---:|---:|----------:|------:|")
    lines.append(f"| {mb['label']} | {mb['total_calls']} | {mb['tp']} | {mb['fp']} | {mb['precision']:.1%} | ${mb['value']:,.2f} |")
    for m in data["mini_models"]:
        lines.append(f"| {m['label']} | {m['total_calls']} | {m['tp']} | {m['fp']} | {m['precision']:.1%} | ${m['value']:,.2f} |")
    lines.append("")

    # Large holdout projection
    lines.append("### 3. Large Holdout Projection\n")
    lines.append("![Large Holdout](campaign_large_holdout.png)\n")
    lb = data["large_baseline"]
    lines.append(f"| Model | Calls | TP (est) | FP (est) | Value (projected) |")
    lines.append(f"|-------|------:|---------:|---------:|------------------:|")
    lines.append(f"| {lb['label']} | {lb['total_calls']:,} | {lb['tp_est']:.0f} | {lb['fp_est']:.0f} | ${lb['value_projected']:,.2f} |")
    for p in data["large_projections"]:
        lines.append(f"| {p['label']} | {p['total_calls']:,} | {p['tp_est']:.0f} | {p['fp_est']:.0f} | ${p['value_projected']:,.2f} |")
    lines.append("")

    return "\n".join(lines)
