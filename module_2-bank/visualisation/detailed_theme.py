"""
Shared theming helpers for detailed charts.
"""

THEMES = {
    "light": {
        "fig_bg": "#ffffff",
        "ax_bg": "#ffffff",
        "text": "#1a2420",
        "text_bold": "#0f1b16",
        "grid": "#d6e0d8",
        "spine": "#60756a",
        "neutral": "#aaaaaa",
        "good": "#2a9d8f",
        "bad": "#e76f51",
        "blue": "#4C72B0",
        "orange": "#DD8452",
        "purple": "#8172B3",
        "baseline": "#e76f51",
    },
    "dark": {
        "fig_bg": "#0f1713",
        "ax_bg": "#17241d",
        "text": "#ecf6f0",
        "text_bold": "#f4fbf7",
        "grid": "#2c3d33",
        "spine": "#3a5044",
        "neutral": "#8fa99c",
        "good": "#54d39a",
        "bad": "#ff8d7a",
        "blue": "#8ab4ff",
        "orange": "#ffb26b",
        "purple": "#c2a7ff",
        "baseline": "#ff8d7a",
    },
}


def get_theme(name: str) -> dict:
    return THEMES.get(name, THEMES["light"])


def apply_figure(fig, theme: dict) -> None:
    fig.patch.set_facecolor(theme["fig_bg"])
    fig.set_facecolor(theme["fig_bg"])


def apply_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["ax_bg"])
    ax.tick_params(colors=theme["text"])
    ax.xaxis.label.set_color(theme["text"])
    ax.yaxis.label.set_color(theme["text"])
    ax.title.set_color(theme["text_bold"])
    for spine in ax.spines.values():
        spine.set_color(theme["spine"])


def style_legend(legend, theme: dict) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(theme["ax_bg"])
    frame.set_edgecolor(theme["spine"])
    for text in legend.get_texts():
        text.set_color(theme["text"])
