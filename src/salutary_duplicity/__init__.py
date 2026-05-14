__version__ = "0.1.0"

from salutary_duplicity.model import SalutaryModel, DEFAULT_PARAMS
from salutary_duplicity.networks import (
    make_erdos_renyi,
    make_small_world,
    make_karate_club,
    make_two_block_sbm,
    assign_random_priors,
    assign_random_strategies,
)

__all__ = [
    "SalutaryModel",
    "DEFAULT_PARAMS",
    "make_erdos_renyi",
    "make_small_world",
    "make_karate_club",
    "make_two_block_sbm",
    "assign_random_priors",
    "assign_random_strategies",
]
