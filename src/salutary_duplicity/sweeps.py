"""
Sweep helpers for high-resolution SBM experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from salutary_duplicity.model import SalutaryModel
from salutary_duplicity.networks import (
    assign_random_priors,
    assign_random_strategies,
    make_karate_club,
    make_two_block_sbm,
)


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for the 100-node SBM parameter sweeps."""

    network_kind: str = "sbm"
    block_sizes: tuple[int, int] = (50, 50)
    p_in: float = 0.18
    p_out: float = 0.04
    sigma_prior: float = 0.5
    n_epochs: int = 30
    burn_in: int = 8
    beta: float = 2.0
    n_sweeps: int = 30
    delta: float = 1.0
    fa: tuple[float, float] | None = None
    fa_total_ratio: float = 1.0
    gamma: float = 2.0
    j_strength: float = 0.5
    individual_learning_cost: float = 0.5
    phase_a_samples: int = 1
    phase_a_reset_mode: str = "random"
    strategy_alpha: float = 0.05
    update_mode: str = "macro"
    base_c_mean: float = 1.0
    alpha_profile: str = "constant"
    alpha_concentration: float = 12.0


DEFAULT_SWEEP_CONFIG = SweepConfig()


def build_network(config: SweepConfig, seed: int) -> Any:
    """Construct the requested social network for a sweep run."""
    if config.network_kind == "sbm":
        return make_two_block_sbm(
            config.block_sizes,
            p_in=config.p_in,
            p_out=config.p_out,
            seed=seed,
        )
    if config.network_kind == "karate":
        return make_karate_club()
    raise ValueError(f"Unknown network_kind: {config.network_kind!r}")


def qualities_from_delta_c(delta_c: float, base_c_mean: float = 1.0) -> tuple[float, float]:
    """Return pasture qualities with fixed mean and gap delta_c."""
    return (base_c_mean + delta_c / 2.0, base_c_mean - delta_c / 2.0)


def forage_availability_from_ratio(
    n_agents: int,
    fa_total_ratio: float,
) -> tuple[float, float]:
    """
    Return symmetric site capacities from forage abundance.

    `fa_total_ratio = 1.0` means total forage equals population size, so each
    site gets capacity `N / 2`. At `2.0`, either site can hold the full
    population while staying below the congestion threshold.
    """
    if fa_total_ratio <= 0.0:
        raise ValueError(f"fa_total_ratio must be positive, got {fa_total_ratio}.")
    per_site = 0.5 * float(fa_total_ratio) * float(n_agents)
    return (per_site, per_site)


def resolve_fa(config: SweepConfig, n_agents: int) -> tuple[float, float]:
    """Resolve the per-site forage capacities for one run."""
    if config.fa is not None:
        return tuple(float(x) for x in config.fa)
    return forage_availability_from_ratio(n_agents=n_agents, fa_total_ratio=config.fa_total_ratio)


def config_to_json(config: SweepConfig) -> str:
    """Serialize a sweep config for metadata storage."""
    return json.dumps(asdict(config), sort_keys=True)


def resolve_alpha_parameter(
    alpha_mean: float,
    n_agents: int,
    seed: int,
    config: SweepConfig = DEFAULT_SWEEP_CONFIG,
) -> float | np.ndarray:
    """
    Resolve the model's alpha parameter from a sweep-level mean alpha value.

    `constant` reproduces the original scalar-alpha model. `beta` draws one
    alpha per agent from a Beta distribution with the requested mean and a
    concentration parameter controlling heterogeneity.
    """
    alpha_mean = float(alpha_mean)
    if not 0.0 <= alpha_mean <= 1.0:
        raise ValueError(f"alpha_mean must be in [0, 1], got {alpha_mean}.")

    profile = str(config.alpha_profile)
    if profile == "constant":
        return alpha_mean

    if profile == "beta":
        concentration = float(config.alpha_concentration)
        if concentration <= 0.0:
            raise ValueError(
                f"alpha_concentration must be positive for beta alpha_profile, got {concentration}."
            )
        if alpha_mean == 0.0:
            return np.zeros(n_agents, dtype=float)
        if alpha_mean == 1.0:
            return np.ones(n_agents, dtype=float)

        rng = np.random.default_rng(seed)
        a = alpha_mean * concentration
        b = (1.0 - alpha_mean) * concentration
        return rng.beta(a, b, size=n_agents).astype(float)

    raise ValueError(f"Unknown alpha_profile: {profile!r}.")


def simulate_run(
    alpha: float,
    p_honest: float,
    delta_c: float,
    seed: int,
    config: SweepConfig = DEFAULT_SWEEP_CONFIG,
) -> dict[str, Any]:
    """
    Run one SBM simulation and return summary paths and post-burn-in statistics.
    """
    G = build_network(config, seed=seed)
    N = G.number_of_nodes()
    h = assign_random_priors(G, sigma=config.sigma_prior, seed=seed + 1_000)
    strategies = assign_random_strategies(N, p_honest=p_honest, seed=seed + 2_000)
    alpha_param = resolve_alpha_parameter(alpha, n_agents=N, seed=seed + 2_500, config=config)
    params = {
        "J_strength": config.j_strength,
        "beta": config.beta,
        "n_sweeps": config.n_sweeps,
        "delta": config.delta,
        "C": list(qualities_from_delta_c(delta_c, base_c_mean=config.base_c_mean)),
        "gamma": config.gamma,
        "alpha": alpha_param,
        "individual_learning_cost": config.individual_learning_cost,
        "phase_a_samples": config.phase_a_samples,
        "phase_a_reset_mode": config.phase_a_reset_mode,
        "strategy_alpha": config.strategy_alpha,
        "update_mode": config.update_mode,
    }
    if config.fa is None:
        params["forage_abundance"] = config.fa_total_ratio
    else:
        params["FA"] = list(resolve_fa(config, n_agents=N))

    model = SalutaryModel(G, h, strategies, params=params, seed=seed + 3_000)
    history = model.run(config.n_epochs)

    realized_fraction_path = np.array([row["fraction_honest"] for row in history], dtype=float)
    latent_fraction_path = np.array([row["latent_fraction_honest"] for row in history], dtype=float)
    reward_path = np.array([row["mean_payoff"] for row in history], dtype=float)
    honest_reward_path = np.array([row["mean_payoff_honest"] for row in history], dtype=float)
    deceptive_reward_path = np.array([row["mean_payoff_deceptive"] for row in history], dtype=float)
    payoff_gap_path = honest_reward_path - deceptive_reward_path

    post_slice = slice(config.burn_in, None)
    post_gap = payoff_gap_path[post_slice]
    post_gap_mean = float("nan") if np.isnan(post_gap).all() else float(np.nanmean(post_gap))

    return {
        "fraction_path": latent_fraction_path,
        "realized_fraction_path": realized_fraction_path,
        "latent_fraction_path": latent_fraction_path,
        "reward_path": reward_path,
        "payoff_gap_path": payoff_gap_path,
        "post_mean_reward": float(np.mean(reward_path[post_slice])),
        "post_mean_payoff_gap": post_gap_mean,
        "final_fraction_honest": float(latent_fraction_path[-1]),
    }


def aggregate_paths(runs: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate repeated-run paths into mean and std trajectories."""
    mat = np.stack([np.asarray(run[key], dtype=float) for run in runs], axis=0)
    return mat.mean(axis=0), mat.std(axis=0)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float("nan") if np.isnan(arr).all() else float(np.nanmean(arr))


def _nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float("nan") if np.isnan(arr).all() else float(np.nanstd(arr))


def summarize_grid_point(
    alpha: float,
    p_honest: float,
    delta_c: float,
    seeds: list[int],
    config: SweepConfig = DEFAULT_SWEEP_CONFIG,
) -> dict[str, float]:
    """Average summary statistics over repeated seeds for one grid point."""
    runs = [simulate_run(alpha, p_honest, delta_c, seed=seed, config=config) for seed in seeds]

    reward_values = [run["post_mean_reward"] for run in runs]
    payoff_gap_values = [run["post_mean_payoff_gap"] for run in runs]
    final_fraction_values = [run["final_fraction_honest"] for run in runs]

    return {
        "reward_mean": float(np.mean(reward_values)),
        "reward_std": float(np.std(reward_values)),
        "payoff_gap_mean": _nanmean(payoff_gap_values),
        "payoff_gap_std": _nanstd(payoff_gap_values),
        "final_fraction_mean": float(np.mean(final_fraction_values)),
        "final_fraction_std": float(np.std(final_fraction_values)),
    }
