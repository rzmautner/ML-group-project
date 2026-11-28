"""Shared matplotlib style so every notebook's charts look the same.

Categorical colors are used in fixed order (never cycled past what's needed);
sequential data uses a single blue ramp.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Validated categorical order: blue, orange, aqua, yellow, magenta, green, violet, red.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[:3]
MUTED = "#a9a8a3"  # for context bars (e.g. "overall" reference)

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("hmda_blue", BLUE_RAMP)


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "figure.figsize": (8, 4.5),
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.markersize": 6,
        "font.size": 10, "text.color": TEXT,
    })


def rate_barh(ax, labels, rates, counts=None, overall=None, title="", xlabel="Denial rate", color=BLUE):
    """Horizontal bars of a rate (0-1), labeled with the value and optional n."""
    y = range(len(labels))
    ax.barh(y, rates, color=color, height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel=xlabel)
    xmax = max(rates) if len(rates) else 1
    for i, r in enumerate(rates):
        txt = f"{r:.0%}" + (f"  (n={counts[i]:,.0f})" if counts is not None else "")
        ax.text(r + xmax * 0.01, i, txt, va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, xmax * 1.35)
    if overall is not None:
        ax.axvline(overall, color=TEXT_SECONDARY, linewidth=1, linestyle="--", label=f"overall {overall:.0%}")
        ax.legend(loc="lower right")
    return ax


def importance_barh(ax, names, values, title="", xlabel="Importance", color=BLUE, fmt="{:.3f}"):
    y = range(len(names))
    ax.barh(y, values, color=color, height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel=xlabel)
    vmax = max(values) if len(values) else 1
    for i, v in enumerate(values):
        ax.text(v + vmax * 0.01, i, fmt.format(v), va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, vmax * 1.18)
    return ax

