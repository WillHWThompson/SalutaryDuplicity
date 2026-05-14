"""
House-style helpers for muted savannah-inspired figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .palette import GRID, INK, WALNUT, WHITE


def apply_anthropology_style() -> None:
    """Apply a minimal, timeless plotting style."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.edgecolor": WALNUT,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "grid.linestyle": "-",
            "font.family": "serif",
            "font.serif": ["Palatino", "Cormorant Garamond", "Georgia", "DejaVu Serif"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": True,
            "legend.facecolor": WHITE,
            "legend.edgecolor": "#d9ccb9",
            "legend.framealpha": 0.95,
        }
    )


def style_axes(ax) -> None:
    """Apply consistent axis styling."""
    ax.set_facecolor(WHITE)
    ax.spines["left"].set_color(WALNUT)
    ax.spines["bottom"].set_color(WALNUT)
    ax.tick_params(colors=INK)
