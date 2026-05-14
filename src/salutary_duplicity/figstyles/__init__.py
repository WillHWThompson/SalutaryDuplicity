"""
Savannah-inspired figure styling for SalutaryDuplicity.
"""

from .palette import fraction_cmap, payoff_gap_cmap, reward_cmap, trajectory_palette
from .style import apply_anthropology_style, style_axes

__all__ = [
    "apply_anthropology_style",
    "style_axes",
    "reward_cmap",
    "fraction_cmap",
    "payoff_gap_cmap",
    "trajectory_palette",
]
