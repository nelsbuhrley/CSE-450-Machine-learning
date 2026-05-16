"""
Value-per-call story chart — the "why we need ML" setup chart.

Single panel showing the bank loses money on mass calls but earns
handsomely on targeted ones. Sets up the ML pitch delivered by
the other marketing charts.

Output: output/marketing/{theme}_bars/value_per_call_story.png (3 themes)

Usage:
    cd visualisation && python may_value_per_call.py
"""

from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from marketing_plots import THEMES, _setup, _theme_ax, _theme_fig, _bar_label

# ── Cost model (matches campaign_analysis.py) ─────────────────
CALL_COST = -11 * 0.5                       # -$5.50 per call
BENEFIT   =  4960 * 0.75 * 0.012            # $44.64 per conversion


def _vpc(conv_rate):
    return CALL_COST + conv_rate * BENEFIT


# ── Data from may_conversion_drop_analysis.md ─────────────────
MAY_CONTACTS    = 12_370
MAY_CONV        = 800
MAY_RATE        = MAY_CONV / MAY_CONTACTS            # 6.5%
MAY_VPC         = _vpc(MAY_RATE)                     # ≈ -$2.61

TOTAL_CONTACTS  = 37_069
TOTAL_CONV      = 252 + 496 + 800 + 502 + 589 + 583 + 236 + 291 + 381 + 78  # 4,208
TOTAL_RATE      = TOTAL_CONV / TOTAL_CONTACTS        # 11.4%
TOTAL_VPC       = _vpc(TOTAL_RATE)                   # ≈ -$0.43

# Targeted months: Mar + Sep + Oct + Dec
TGT_CONTACTS    = 496 + 508 + 653 + 158              # 1,815
TGT_CONV        = 252 + 236 + 291 + 78               # 857
TGT_RATE        = TGT_CONV / TGT_CONTACTS            # 47.2%
TGT_VPC         = _vpc(TGT_RATE)                     # ≈ $15.58


# ── Chart ─────────────────────────────────────────────────────

def make_chart(theme_name):
    _setup()
    t = THEMES[theme_name]

    fig, ax = plt.subplots(figsize=(8, 7))

    _theme_fig(fig, t,
               "Every Call Has a Price Tag",
               "We already know targeted campaigns work. The question is how to scale it.")
    _theme_ax(ax, t)

    labels = ["May\nmass calls", "Campaign\naverage", "Targeted\nmonths*"]
    vals   = [MAY_VPC, TOTAL_VPC, TGT_VPC]
    cols   = [t["bad"], t["bad"], t["good"]]

    ax.bar(labels, vals, color=cols,
           edgecolor=t["bar_edge"], width=0.52, zorder=3)
    ax.axhline(0, color=t["subtle"], linewidth=0.8, zorder=2)

    span = max(abs(MAY_VPC), TGT_VPC)
    pad  = span * 0.08

    # Dollar labels
    _bar_label(ax, 0, MAY_VPC   - pad, f"${MAY_VPC:.2f}",
               t["bad"], above=False)
    _bar_label(ax, 1, TOTAL_VPC - pad, f"${TOTAL_VPC:.2f}",
               t["bad"], above=False)
    _bar_label(ax, 2, TGT_VPC   + pad, f"${TGT_VPC:.2f}",
               t["good"], above=True)

    # Volume & rate annotations
    ann_fs = 9.5
    ax.text(0, MAY_VPC - pad * 3.6,
            f"{MAY_CONTACTS:,} calls · {MAY_RATE:.1%} convert",
            ha="center", fontsize=ann_fs, color=t["subtle"], style="italic")
    ax.text(1, TOTAL_VPC - pad * 3.6,
            f"{TOTAL_CONTACTS:,} calls · {TOTAL_RATE:.1%} convert",
            ha="center", fontsize=ann_fs, color=t["subtle"], style="italic")
    ax.text(2, TGT_VPC + pad * 3.4,
            f"{TGT_CONTACTS:,} calls · {TGT_RATE:.1%} convert",
            ha="center", fontsize=ann_fs, color=t["subtle"], style="italic")

    ax.set_title("The Spray-and-Pray Tax",
                 color=t["text_bold"], fontsize=15, fontweight="bold", pad=20)
    ax.set_ylabel("Value per Call ($)", color=t["text"], fontsize=13)
    ax.grid(axis="y", color=t["grid"], zorder=0, linewidth=0.6)
    ax.set_ylim(MAY_VPC - span * 0.50, TGT_VPC + span * 0.45)

    # Break-even arrow
    ax.annotate("break-even", xy=(1.55, 0), xytext=(1.55, -span * 0.12),
                fontsize=9, color=t["subtle"], style="italic", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=t["subtle"],
                                lw=0.8, shrinkA=0, shrinkB=2))

    # Footnote
    ax.text(0.5, 0.02, "* Mar, Sep, Oct, Dec — small, warm-lead campaigns",
            transform=ax.transAxes, fontsize=9, color=t["subtle"],
            style="italic", ha="center")

    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.10, top=0.80)
    return fig


# ── Generator ─────────────────────────────────────────────────

def generate_all():
    _setup()
    base = Path(__file__).parent / "output" / "marketing"

    for theme in THEMES:
        folder = base / f"{theme}_bars"
        folder.mkdir(parents=True, exist_ok=True)
        fig = make_chart(theme)
        out = folder / "value_per_call_story.png"
        fig.savefig(out, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: {out.relative_to(base)}")

    print("Done — value_per_call_story.png generated in all 3 themes.")


if __name__ == "__main__":
    generate_all()
