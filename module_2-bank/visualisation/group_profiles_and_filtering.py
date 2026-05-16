"""
Group profile visualisations:
1. Top-3 groups most/least likely to subscribe
2. How each model filters those groups vs calling everyone

Called by main_vis.py — not intended to run standalone.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent
TRAIN_PATH = ROOT / "training_data" / "bank.csv"
TEST_PATH = ROOT / "test_data" / "bank_holdout_test_mini.csv"
ANSWERS_PATH = ROOT / "verification_data" / "bank_holdout_test_mini_answers.csv"
PRED_DIR = ROOT / "predictions" / "mini-holdout"

GREEN = "#2a9d8f"
RED = "#e76f51"
BLUE = "#4C72B0"
ORANGE = "#DD8452"
PURPLE = "#8172B3"
GREY = "#aaaaaa"

CATS = ["job", "marital", "education", "contact", "poutcome"]

MODEL_SHORT_NAMES = {
    "1_caleb_rf": "Model 1 (Caleb RF)",
    "2_caleb_rf": "Model 2 (Caleb RF*)",
    "3_nels_stack_rf_knn": "Model 3 (Nels Stack)",
}


def _conversion_rates(df, col):
    return df.groupby(col)["y"].apply(lambda x: (x == "yes").mean()).sort_values(ascending=False)


def _load_mini_holdout():
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    answers = pd.read_csv(ANSWERS_PATH)
    test["y_actual"] = answers.iloc[:, 0]
    keys = []
    for f in sorted(PRED_DIR.glob("*.csv")):
        stem = f.stem.removeprefix("NorthWindModule2_")
        key = stem.rsplit("_mini-holdout-predictions", 1)[0]
        test[f"pred_{key}"] = pd.read_csv(f).iloc[:, 0].values
        keys.append(key)
    return test, keys


def _get_top_bottom_groups(train):
    rows = []
    for col in CATS:
        rates = _conversion_rates(train, col)
        counts = train[col].value_counts()
        for val, rate in rates.items():
            n = counts[val]
            if n < 30:
                continue
            rows.append({"feature": col, "value": val, "rate": rate, "n": n})
    all_groups = pd.DataFrame(rows)
    top3 = all_groups.nlargest(3, "rate")
    bot3 = all_groups.nsmallest(3, "rate")
    return top3, bot3


def run_analysis() -> dict:
    train = pd.read_csv(TRAIN_PATH)
    test, model_keys = _load_mini_holdout()
    baseline = (train["y"] == "yes").mean()
    top3, bot3 = _get_top_bottom_groups(train)

    # Filtering proportions
    groups = list(zip(top3["feature"], top3["value"])) + list(zip(bot3["feature"], bot3["value"]))
    group_labels = [f"{f}: {v}" for f, v in groups]
    everyone_pcts = [(test[f] == v).mean() * 100 for f, v in groups]
    model_pcts = {}
    for key in model_keys:
        called = test[test[f"pred_{key}"] == 1]
        pcts = [(called[f] == v).mean() * 100 if len(called) > 0 else 0 for f, v in groups]
        model_pcts[key] = pcts

    return {
        "baseline_rate": baseline,
        "top3": top3.to_dict("records"),
        "bot3": bot3.to_dict("records"),
        "group_labels": group_labels,
        "everyone_pcts": everyone_pcts,
        "model_pcts": model_pcts,
        "model_keys": model_keys,
        "_train": train, "_test": test,
    }


def make_figures(data: dict) -> dict[str, plt.Figure]:
    figs = {}

    # --- Fig 1: group profiles ---
    baseline = data["baseline_rate"]
    top3 = data["top3"]
    bot3 = data["bot3"]

    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(13, 5))

    labels_t = [f"{r['feature']}: {r['value']}\n(n={r['n']:,})" for r in top3]
    rates_t = [r["rate"] * 100 for r in top3]
    bars_t = ax_top.barh(labels_t[::-1], rates_t[::-1], color=GREEN, edgecolor="white", height=0.55)
    ax_top.axvline(baseline * 100, color=GREY, linestyle="--", linewidth=1, label=f"baseline {baseline*100:.1f}%")
    for bar, rate in zip(bars_t, rates_t[::-1]):
        ax_top.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.1f}%", va="center", fontsize=11, fontweight="bold", color=GREEN)
    ax_top.set_xlim(0, max(rates_t) * 1.25)
    ax_top.set_xlabel("Conversion rate (%)")
    ax_top.set_title("Top 3 most likely to subscribe", fontsize=13, fontweight="bold", color=GREEN)
    ax_top.legend(fontsize=9)
    ax_top.spines[["top", "right"]].set_visible(False)

    labels_b = [f"{r['feature']}: {r['value']}\n(n={r['n']:,})" for r in bot3]
    rates_b = [r["rate"] * 100 for r in bot3]
    bars_b = ax_bot.barh(labels_b[::-1], rates_b[::-1], color=RED, edgecolor="white", height=0.55)
    ax_bot.axvline(baseline * 100, color=GREY, linestyle="--", linewidth=1, label=f"baseline {baseline*100:.1f}%")
    for bar, rate in zip(bars_b, rates_b[::-1]):
        ax_bot.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.1f}%", va="center", fontsize=11, fontweight="bold", color=RED)
    ax_bot.set_xlim(0, baseline * 100 * 1.6)
    ax_bot.set_xlabel("Conversion rate (%)")
    ax_bot.set_title("Top 3 least likely to subscribe", fontsize=13, fontweight="bold", color=RED)
    ax_bot.legend(fontsize=9)
    ax_bot.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Who says YES and who says NO?  (training data, n >= 30)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    figs["group_profiles"] = fig

    # --- Fig 2: filtering effect ---
    group_labels = data["group_labels"]
    everyone_pcts = data["everyone_pcts"]
    model_pcts = data["model_pcts"]
    model_keys = data["model_keys"]

    n_groups = len(group_labels)
    n_bars = 1 + len(model_keys)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig2, ax = plt.subplots(figsize=(14, 6.5))
    ax.bar(x - (n_bars - 1) * bar_width / 2, everyone_pcts, bar_width,
           label="Call everyone", color=GREY, edgecolor="white", zorder=3)
    colors = [BLUE, ORANGE, PURPLE]
    for i, key in enumerate(model_keys):
        offset = (i + 1 - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, model_pcts[key], bar_width,
               label=MODEL_SHORT_NAMES.get(key, key), color=colors[i % len(colors)],
               edgecolor="white", zorder=3)

    ax.axvline(2.5, color="#cccccc", linestyle="-", linewidth=1.5, zorder=1)
    ylim = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 30
    ax.text(1.0, ylim, "HIGH conversion", ha="center", fontsize=10, fontweight="bold", color=GREEN, style="italic")
    ax.text(4.5, ylim, "LOW conversion", ha="center", fontsize=10, fontweight="bold", color=RED, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("% of people called", fontsize=11)
    ax.set_title("Model filtering: proportion of key groups in call list vs calling everyone\n(mini-holdout)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    all_vals = everyone_pcts + [v for pcts in model_pcts.values() for v in pcts]
    ax.set_ylim(0, max(all_vals) * 1.3)
    fig2.tight_layout()
    figs["model_filtering"] = fig2

    return figs


def make_markdown(data: dict) -> str:
    lines = []
    baseline = data["baseline_rate"]
    lines.append("## Group Profiles\n")
    lines.append(f"Overall baseline conversion rate: **{baseline:.1%}**\n")

    lines.append("### Most likely to subscribe\n")
    lines.append("![Group Profiles](group_profiles.png)\n")
    lines.append("| Group | Conversion Rate | Sample Size |")
    lines.append("|-------|----------------:|------------:|")
    for r in data["top3"]:
        lines.append(f"| {r['feature']}: {r['value']} | {r['rate']:.1%} | {r['n']:,} |")
    lines.append("")

    lines.append("### Least likely to subscribe\n")
    lines.append("| Group | Conversion Rate | Sample Size |")
    lines.append("|-------|----------------:|------------:|")
    for r in data["bot3"]:
        lines.append(f"| {r['feature']}: {r['value']} | {r['rate']:.1%} | {r['n']:,} |")
    lines.append("")

    lines.append("### Model Filtering Effect\n")
    lines.append("![Model Filtering](model_filtering.png)\n")
    lines.append("How much of each group ends up in the call list (mini-holdout):\n")
    header = "| Group | Everyone |"
    sep = "|-------|--------:|"
    for key in data["model_keys"]:
        name = MODEL_SHORT_NAMES.get(key, key)
        header += f" {name} |"
        sep += "--------:|"
    lines.append(header)
    lines.append(sep)
    for i, label in enumerate(data["group_labels"]):
        row = f"| {label} | {data['everyone_pcts'][i]:.1f}% |"
        for key in data["model_keys"]:
            row += f" {data['model_pcts'][key][i]:.1f}% |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)
