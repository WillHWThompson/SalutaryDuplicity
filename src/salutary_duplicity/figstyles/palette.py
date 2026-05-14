"""
Savannah-inspired palette definitions for figures.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

WHITE = "#ffffff"
PARCHMENT = "#f7f1e3"
PAPER = "#fcf8ef"
INK = "#41362f"
WALNUT = "#6a5847"
SAND = "#d6c3a2"
DRY_GRASS = "#b8c3a1"
ACACIA = "#93a37d"
DUSTY_SKY = "#9eb6bd"
CLAY = "#c68f74"
RIVERSTONE = "#7f919b"
ROSE_DUST = "#c7a298"
GRID = "#ddd2bf"


def reward_cmap() -> LinearSegmentedColormap:
    """Sequential colormap for payoff surfaces."""
    return LinearSegmentedColormap.from_list(
        "savannah_reward",
        [WALNUT, CLAY, SAND, DRY_GRASS, "#dfe7cf"],
    )


def fraction_cmap() -> LinearSegmentedColormap:
    """Sequential colormap for replicator outcomes."""
    return LinearSegmentedColormap.from_list(
        "savannah_fraction",
        ["#2f2722", WALNUT, CLAY, ROSE_DUST, PARCHMENT],
    )


def payoff_gap_cmap() -> LinearSegmentedColormap:
    """Diverging colormap for honest-minus-deceptive payoff gaps."""
    return LinearSegmentedColormap.from_list(
        "savannah_gap",
        [DUSTY_SKY, "#dce5e3", PARCHMENT, "#ead7cf", CLAY],
    )


def trajectory_palette(n: int) -> list[str]:
    """Muted line colors for trajectory overlays."""
    base = [DUSTY_SKY, CLAY, ACACIA, ROSE_DUST, RIVERSTONE, SAND]
    return [base[i % len(base)] for i in range(n)]
