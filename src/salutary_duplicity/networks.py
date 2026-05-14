"""
Network construction and initialization helpers for SalutaryDuplicity.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def make_erdos_renyi(N: int, p: float, seed: int | None = None) -> nx.Graph:
    """
    Erdos-Renyi random graph G(N, p).

    Parameters
    ----------
    N : int
        Number of nodes.
    p : float
        Edge probability.
    seed : int or None
        RNG seed.

    Returns
    -------
    nx.Graph
        Graph with integer nodes 0..N-1.
    """
    return nx.erdos_renyi_graph(N, p, seed=seed)


def make_small_world(N: int, k: int, p: float, seed: int | None = None) -> nx.Graph:
    """
    Watts-Strogatz small-world graph.

    Parameters
    ----------
    N : int
        Number of nodes.
    k : int
        Each node is initially connected to k nearest neighbors in ring.
    p : float
        Probability of rewiring each edge.
    seed : int or None
        RNG seed.

    Returns
    -------
    nx.Graph
    """
    return nx.watts_strogatz_graph(N, k, p, seed=seed)


def make_karate_club() -> nx.Graph:
    """
    Zachary's karate club graph (34 nodes, 78 edges). Good for quick tests.

    Returns
    -------
    nx.Graph
        Relabeled to integer nodes 0..33.
    """
    G = nx.karate_club_graph()
    # Ensure nodes are 0-indexed integers (they already are, but be explicit)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G


def make_two_block_sbm(
    block_sizes: tuple[int, int] | list[int],
    p_in: float,
    p_out: float,
    seed: int | None = None,
) -> nx.Graph:
    """
    Two-community stochastic block model.

    Parameters
    ----------
    block_sizes : tuple[int, int] or list[int]
        Sizes of the two blocks.
    p_in : float
        Within-block edge probability.
    p_out : float
        Between-block edge probability.
    seed : int or None
        RNG seed.

    Returns
    -------
    nx.Graph
        Graph with integer nodes 0..N-1.
    """
    sizes = list(block_sizes)
    if len(sizes) != 2:
        raise ValueError(f"block_sizes must have length 2, got {sizes!r}.")

    probs = [
        [p_in, p_out],
        [p_out, p_in],
    ]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return nx.convert_node_labels_to_integers(G, first_label=0)


def assign_random_priors(
    G: nx.Graph,
    sigma: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Assign private priors h_i ~ N(0, sigma^2) to each agent.

    Parameters
    ----------
    G : nx.Graph
        Social network (used only for its size N = number of nodes).
    sigma : float
        Standard deviation of the prior distribution.
    seed : int or None
        RNG seed.

    Returns
    -------
    np.ndarray, shape (N,)
        Private prior values. Positive = leans toward site +1,
        negative = leans toward site -1.
    """
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    return rng.normal(loc=0.0, scale=sigma, size=N)


def assign_random_strategies(
    N: int,
    p_honest: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Randomly assign Honest/Deceptive strategies to N agents.

    Parameters
    ----------
    N : int
        Number of agents.
    p_honest : float
        Probability that each agent starts Honest.
    seed : int or None
        RNG seed.

    Returns
    -------
    np.ndarray, shape (N,), dtype bool
        True = Honest, False = Deceptive.
    """
    rng = np.random.default_rng(seed)
    return rng.random(N) < p_honest
