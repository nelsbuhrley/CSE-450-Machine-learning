"""
Marketing-grade C-suite presentation charts.

Generates two key visuals across 3 themes:
  1. Campaign Value — 2-panel proof (exact) + projection (with error bars)
  2. The Golden List & Black List — conversion rates + model filtering (bars)

Output: output/marketing/{theme}_bars/ with 2 PNGs each.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Themes ──────────────────────────────────────────────────

THEMES = {
    "dark": {
        "fig_bg":    "#0d1117",
        "ax_bg":     "#161b22",
        "text":      "#b0b8c4",
        "text_bold": "#f0f6fc",
        "good":      "#3fb9a8",
        "bad":       "#ff6b6b",
        "neutral":   "#6e7681",
        "grid":      "#21262d",
        "subtle":    "#8b949e",
        "bar_edge":  "#0d1117",
        "spine":     "#30363d",
        "error_bar": "#e6edf3",
        "baseline":  "#f0b866",
    },
    "white": {
        "fig_bg":    "#ffffff",
        "ax_bg":     "#ffffff",
        "text":      "#57606a",
        "text_bold": "#1a1a2e",
        "good":      "#1a7f72",
        "bad":       "#cf4e3e",
        "neutral":   "#a0a8b2",
        "grid":      "#f6f8fa",
        "subtle":    "#8b949e",
        "bar_edge":  "#ffffff",
        "spine":     "#d8dee4",
        "error_bar": "#24292f",
        "baseline":  "#d4880f",
    },
    "grey": {
        "fig_bg":    "#f0f2f5",
        "ax_bg":     "#ffffff",
        "text":      "#57606a",
        "text_bold": "#1a1a2e",
        "good":      "#2a9d8f",
        "bad":       "#e76f51",
        "neutral":   "#9ca3ad",
        "grid":      "#eaecef",
        "subtle":    "#8b949e",
        "bar_edge":  "#f0f2f5",
        "spine":     "#d8dee4",
        "error_bar": "#24292f",
        "baseline":  "#c77b10",
    },
}


# ── Styling ─────────────────────────────────────────────────

def _setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "font.size": 11,
        "axes.linewidth": 0.8,
    })


def _theme_ax(ax, t):
    ax.set_facecolor(t["ax_bg"])
    ax.tick_params(colors=t["text"], labelsize=12, length=4)
    for spine in ax.spines.values():
        spine.set_color(t["spine"])
    ax.spines[["top", "right"]].set_visible(False)


def _theme_fig(fig, t, title, subtitle):
    fig.set_facecolor(t["fig_bg"])
    fig.suptitle(title, fontsize=20, fontweight="bold", color=t["text_bold"],
                 fontfamily="sans-serif", y=0.99)
    fig.text(0.5, 0.94, subtitle, ha="center", va="top",
             fontsize=14, color=t["subtle"], style="italic")


def _bar_label(ax, x, y, text, color, above=True, fontsize=14):
    va = "bottom" if above else "top"
    ax.text(x, y, text, ha="center", va=va, fontsize=fontsize,
            fontweight="bold", color=color)


# ── Chart 1: Campaign Value ─────────────────────────────────

def make_campaign(campaign_data, theme_name):
    _setup()
    t = THEMES[theme_name]

    # Best model for left panel (exact, no error bar)
    best_mini = max(campaign_data["mini_models"], key=lambda m: m["value"])
    mini_bl = campaign_data["mini_baseline"]["value"]
    mini_best = best_mini["value"]

    # Right panel: projection with error bars from all 3 models
    large_bl = campaign_data["large_baseline"]["value_projected"]
    large_vals = sorted([p["value_projected"] for p in campaign_data["large_projections"]])
    large_mean = np.mean(large_vals)
    large_lo, large_hi = large_vals[0], large_vals[-1]

    mini_n = campaign_data["mini_baseline"]["total_calls"]
    large_n = campaign_data["large_baseline"]["total_calls"]

    title = "Stop Losing Money — Start Calling Smarter"
    subtitle = "Our ML model turns a losing campaign into a profitable one"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7),
                                    gridspec_kw={"wspace": 0.35})
    _theme_fig(fig, t, title, subtitle)
    _theme_ax(ax1, t)
    _theme_ax(ax2, t)

    bar_labels = ["Without ML\nfiltering", "With our\nmodel"]
    bar_colors = [t["bad"], t["good"]]

    # ─ Left panel: proven results (exact, no error bar) ─
    vals1 = [mini_bl, mini_best]
    ax1.bar(bar_labels, vals1, color=bar_colors,
            edgecolor=t["bar_edge"], width=0.55, zorder=3)

    span1 = max(abs(mini_bl), mini_best)
    pad1 = span1 * 0.10
    _bar_label(ax1, 0, mini_bl - pad1, f"${mini_bl:,.0f}",
               t["bad"], above=False)
    _bar_label(ax1, 1, mini_best + pad1, f"${mini_best:,.0f}",
               t["good"], above=True)

    ax1.axhline(0, color=t["subtle"], linewidth=0.8, zorder=2)
    ax1.set_title(f"Here's What You Would Have Had\n({mini_n}-contact test)",
                  color=t["text_bold"], fontsize=13, fontweight="bold", pad=20)
    ax1.set_ylabel("Campaign Value ($)", color=t["text"], fontsize=13)
    ax1.grid(axis="y", color=t["grid"], zorder=0, linewidth=0.6)
    ax1.set_ylim(mini_bl - span1 * 0.35, mini_best + span1 * 0.35)

    # ─ Right panel: projected at scale (with error bars) ─
    vals2 = [large_bl, large_mean]
    ax2.bar(bar_labels, vals2, color=bar_colors,
            edgecolor=t["bar_edge"], width=0.55, zorder=3)
    ax2.errorbar(1, large_mean,
                 yerr=[[large_mean - large_lo], [large_hi - large_mean]],
                 fmt="none", capsize=10, capthick=2.5,
                 ecolor=t["error_bar"], elinewidth=2.5, zorder=4)

    span2 = max(abs(large_bl), large_hi)
    pad2 = span2 * 0.08
    _bar_label(ax2, 0, large_bl - pad2, f"${large_bl:,.0f}",
               t["bad"], above=False)
    _bar_label(ax2, 1, large_hi + pad2, f"${large_mean:,.0f}",
               t["good"], above=True)

    ax2.axhline(0, color=t["subtle"], linewidth=0.8, zorder=2)
    ax2.set_title(f"Here's What You're Likely to Have\n({large_n:,}-contact campaign)",
                  color=t["text_bold"], fontsize=13, fontweight="bold", pad=20)
    ax2.set_ylabel("Projected Value ($)", color=t["text"], fontsize=13)
    ax2.grid(axis="y", color=t["grid"], zorder=0, linewidth=0.6)
    ax2.set_ylim(large_bl - span2 * 0.30, large_hi + span2 * 0.30)

    ax2.text(1, large_hi + span2 * 0.18,
             "range across\n3 model variants",
             ha="center", fontsize=11, color=t["subtle"], style="italic")

    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.06, top=0.80, wspace=0.35)
    return fig, subtitle


# ── Chart 2: Golden/Black List + Filtering ───────────────────

Y_GOLDEN = [5, 4, 3]
Y_BLACK = [1.2, 0.2, -0.8]
Y_ALL = Y_GOLDEN + Y_BLACK

# Natural-language labels for feature:value combos
FRIENDLY_LABELS = {
    ("poutcome", "success"):      "Previously converted",
    ("job", "student"):           "Students",
    ("job", "retired"):           "Retirees",
    ("contact", "telephone"):     "Reached by landline",
    ("job", "blue-collar"):       "Blue-collar workers",
    ("education", "basic.9y"):    "Basic education (9yr)",
}


def _friendly(feature, value):
    return FRIENDLY_LABELS.get((feature, value), f"{feature}: {value}")


def _rates_panel(ax, top3, bot3, baseline, t):
    """Left panel: horizontal conversion rate bars."""
    rates_g = [r["rate"] * 100 for r in top3]
    rates_b = [r["rate"] * 100 for r in bot3]
    labels_g = [_friendly(r["feature"], r["value"]) for r in top3]
    labels_b = [_friendly(r["feature"], r["value"]) for r in bot3]

    bg_box = dict(facecolor=t["ax_bg"], edgecolor="none", pad=1.5)

    ax.barh(Y_GOLDEN, rates_g, color=t["good"], edgecolor=t["bar_edge"],
            height=0.65, zorder=3, alpha=0.90)
    ax.barh(Y_BLACK, rates_b, color=t["bad"], edgecolor=t["bar_edge"],
            height=0.65, zorder=3, alpha=0.90)

    # Baseline — prominent dashed line with label at top
    bl_x = baseline * 100
    ax.axvline(bl_x, color=t["baseline"], linestyle="--",
               linewidth=1.8, zorder=2, alpha=0.85)
    ax.text(bl_x, max(Y_GOLDEN) + 1.35,
            f"baseline {bl_x:.1f}%", fontsize=11, fontweight="bold",
            color=t["baseline"], ha="center", va="bottom",
            bbox=bg_box, zorder=5)

    # Rate labels — with background so baseline line doesn't cut through
    all_y = Y_GOLDEN + Y_BLACK
    all_rates = rates_g + rates_b
    all_cols = [t["good"]] * 3 + [t["bad"]] * 3
    for y, rate, col in zip(all_y, all_rates, all_cols):
        ax.text(rate + 1.2, y, f"{rate:.1f}%", va="center",
                fontsize=13, fontweight="bold", color=col,
                bbox=bg_box, zorder=5)

    all_labels = labels_g + labels_b
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=12, color=t["text"])
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # Section headers — centered in the panel
    xmax = max(rates_g) * 1.30
    xmid = xmax / 2
    ax.text(xmid, max(Y_GOLDEN) + 0.85, "THE GOLDEN LIST",
            fontsize=13, fontweight="bold", color=t["good"],
            va="bottom", ha="center")
    ax.text(xmid, max(Y_GOLDEN) + 0.55, "groups you should call",
            fontsize=11, color=t["subtle"], va="bottom", ha="center",
            style="italic")

    ax.text(xmid, max(Y_BLACK) + 0.85, "THE BLACK LIST",
            fontsize=13, fontweight="bold", color=t["bad"],
            va="bottom", ha="center")
    ax.text(xmid, max(Y_BLACK) + 0.55, "groups you should skip",
            fontsize=11, color=t["subtle"], va="bottom", ha="center",
            style="italic")

    ax.set_xlabel("Conversion Rate (%)", color=t["text"], fontsize=12)
    ax.set_xlim(0, xmax)
    ax.set_ylim(min(Y_BLACK) - 0.8, max(Y_GOLDEN) + 1.5)


def _filter_bars(ax, everyone, model, t, is_golden):
    """Right panel: grouped bars — without ML vs with our model."""
    bh = 0.28
    for i, y in enumerate(Y_ALL):
        ev, ml = everyone[i], model[i]
        good_dir = (ml > ev) if is_golden[i] else (ml < ev)
        color = t["good"] if good_dir else t["bad"]
        ax.barh(y + bh / 2 + 0.02, ev, height=bh, color=t["neutral"],
                edgecolor=t["bar_edge"], zorder=3)
        ax.barh(y - bh / 2 - 0.02, ml, height=bh, color=color,
                edgecolor=t["bar_edge"], zorder=3)
        ax.text(ev + 0.4, y + bh / 2 + 0.02, f"{ev:.0f}%",
                va="center", fontsize=10, color=t["subtle"])
        ax.text(ml + 0.4, y - bh / 2 - 0.02, f"{ml:.0f}%",
                va="center", fontsize=11, fontweight="bold", color=color)


def make_golden_list(gp_data, best_key, theme_name):
    _setup()
    t = THEMES[theme_name]

    top3 = gp_data["top3"]
    bot3 = gp_data["bot3"]
    baseline = gp_data["baseline_rate"]
    everyone = gp_data["everyone_pcts"]
    model = gp_data["model_pcts"][best_key]
    is_golden = [True, True, True, False, False, False]

    title = "Know Who to Call — And Who to Skip"
    subtitle = "Our model concentrates your budget on high-conversion groups and eliminates waste"

    fig, (ax_rates, ax_filter) = plt.subplots(
        1, 2, figsize=(17, 8.5),
        gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.06})
    _theme_fig(fig, t, title, subtitle)
    _theme_ax(ax_rates, t)
    _theme_ax(ax_filter, t)

    # Left: conversion rates
    _rates_panel(ax_rates, top3, bot3, baseline, t)

    # Right: filtering bars
    _filter_bars(ax_filter, everyone, model, t, is_golden)
    ax_filter.set_yticks(Y_ALL)
    ax_filter.set_yticklabels([])
    ax_filter.yaxis.set_minor_locator(plt.NullLocator())
    ax_filter.set_ylim(ax_rates.get_ylim())
    ax_filter.set_xlabel("% of Call List", color=t["text"], fontsize=12)
    xmax = max(max(everyone), max(model)) * 1.45
    ax_filter.set_xlim(0, xmax)
    ax_filter.grid(axis="x", color=t["grid"], zorder=0, linewidth=0.6)

    # Legend
    elements = [
        Patch(facecolor=t["neutral"], edgecolor=t["bar_edge"],
              label="Without ML filtering"),
        Patch(facecolor=t["good"], edgecolor=t["bar_edge"],
              label="With our model"),
    ]
    ax_filter.legend(handles=elements, fontsize=11, loc="lower right",
                     facecolor=t["ax_bg"], edgecolor=t["spine"],
                     labelcolor=t["text"], framealpha=0.9,
                     handlelength=2.5, handleheight=1.5, borderpad=0.8)

    ax_rates.set_title("Why — Conversion Rate", color=t["text_bold"],
                       fontsize=14, fontweight="bold", pad=16)
    ax_filter.set_title("What We Do — Reshape Your Call List",
                        color=t["text_bold"], fontsize=14, fontweight="bold", pad=16)

    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.06, top=0.84, wspace=0.08)
    return fig, subtitle


# ── Generator ───────────────────────────────────────────────

def generate_all(campaign_data, gp_data, best_key, base_dir):
    _setup()
    base_dir.mkdir(parents=True, exist_ok=True)

    for theme_name in THEMES:
        folder = base_dir / f"{theme_name}_bars"
        folder.mkdir(parents=True, exist_ok=True)

        fig_c, sub_c = make_campaign(campaign_data, theme_name)
        fig_c.savefig(
            folder / "campaign_value.png", dpi=200,
            bbox_inches="tight", facecolor=fig_c.get_facecolor(),
            metadata={"Description": sub_c, "Title": "Campaign Value"})
        plt.close(fig_c)

        fig_g, sub_g = make_golden_list(gp_data, best_key, theme_name)
        fig_g.savefig(
            folder / "golden_blacklist.png", dpi=200,
            bbox_inches="tight", facecolor=fig_g.get_facecolor(),
            metadata={"Description": sub_g, "Title": "Golden & Black List"})
        plt.close(fig_g)

        print(f"  Saved: marketing/{theme_name}_bars/")
