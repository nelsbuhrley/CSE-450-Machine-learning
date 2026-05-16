"""
Main visualisation runner.

Generates two sets of output:
  output/detailed/                  — full multi-model analysis
  output/marketing/{theme}_{style}/ — C-suite presentation variants
                                      (3 themes x 3 filter styles = 9 folders)

Usage:
    python visualisation/main_vis.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import campaign_analysis
import grade_value_per_call
import group_profiles_and_filtering
import marketing_plots

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
DETAILED_DIR = OUTPUT_DIR / "detailed"
MARKETING_DIR = OUTPUT_DIR / "marketing"


def main():
    DETAILED_DIR.mkdir(parents=True, exist_ok=True)

    # ── run all analyses once ──
    print("Running analyses...")
    gp_data = group_profiles_and_filtering.run_analysis()
    vpc_data = grade_value_per_call.run_analysis()
    campaign_data = campaign_analysis.run_analysis()

    best = max(campaign_data["mini_models"], key=lambda m: m["value"])
    best_key = best["key"]
    print(f"Best model: {best['label']} (${best['value']:,.2f})\n")

    # ── detailed output ──
    print("=== Detailed charts ===")
    md_sections = ["# Bank Marketing ML Analysis Report\n"]
    modules_data = [
        ("Group Profiles & Filtering", group_profiles_and_filtering, gp_data),
        ("Value per Call", grade_value_per_call, vpc_data),
        ("Campaign Analysis", campaign_analysis, campaign_data),
    ]
    for section_name, mod, data in modules_data:
        figures = mod.make_figures(data)
        for fig_name, fig in figures.items():
            path = DETAILED_DIR / f"{fig_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: detailed/{fig_name}.png")
        md_sections.append(mod.make_markdown(data))

    (DETAILED_DIR / "report.md").write_text("\n".join(md_sections))
    print("  Saved: detailed/report.md")

    # ── marketing output (9 variants) ──
    print("\n=== Marketing charts (3 themes x 3 filter styles) ===")
    marketing_plots.generate_all(campaign_data, gp_data, best_key, MARKETING_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
