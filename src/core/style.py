"""Shared figure style: one palette, recessive axes, direct labels.

Series colors are the first three slots of a CVD-validated categorical order
(blue, orange, aqua); they pass all-pairs separation in light mode. Text stays in
neutral ink, never in a series color.
"""
from __future__ import annotations

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f1f0ec"
SURFACE = "#ffffff"


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9, "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6, "text.color": INK,
    })


def headline(fig: plt.Figure, title: str, subtitle: str, note: str) -> None:
    """Title, wrapped subtitle, and wrapped source note, positioned in inches from the edges."""
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.text(0.01, 1 - 0.08 / h, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.36 / h, textwrap.fill(subtitle, int(w * 15.5)), ha="left", va="top", fontsize=9, color=INK2)
    fig.text(0.01, 0.06 / h, textwrap.fill(note, int(w * 20)), ha="left", va="bottom", fontsize=7, color=MUTED)


def label(ax: plt.Axes, x: float, y: float, text: str, dx: float, dy: float, **kw) -> None:
    """Annotation with a thin leader line, text in neutral ink."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), **kw)
