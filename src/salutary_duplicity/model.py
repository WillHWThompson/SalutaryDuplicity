"""
SalutaryModel: multi-scale pastoral cooperation simulation.

Three-phase per-epoch structure:
  A) Fast timescale -- Ising/Glauber dynamics (daily decisions)
  B) Ecological evaluation -- congestion payoffs
  C) Slow timescale -- strategy update (macro replicator or micro imitation)
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

DEFAULT_PARAMS: dict[str, Any] = {
    "J_strength": 0.5,
    "beta": 1.0,
    "n_sweeps": 50,
    "delta": 1.0,
    "FA": [10, 10],  # carrying capacities [site1, site2]
    "forage_abundance": None,  # total forage / population size; resolves symmetric FA if set
    "C": [1.0, 1.0],  # baseline productivity [site1, site2]
    "gamma": 2.0,
    "alpha": 0.0,  # scalar or per-agent social-vs-individual learning interpolation
    "individual_learning_cost": 1.0,  # payoff penalty coefficient for alpha
    "phase_a_samples": 1,  # average payoffs over multiple Phase A/B realizations
    "phase_a_reset_mode": "random",  # "random" or "current" when phase_a_samples > 1
    "strategy_alpha": 0.05,  # replicator learning rate
    "eta": 0.1,  # imitation scaling
    "macro_assignment_mode": "minimal",  # "minimal" or "resample_all"
    "update_mode": "macro",  # "macro", "micro", "finite", or "none"
}


class SalutaryModel:
    """
    Multi-scale pastoral cooperation model.

    Parameters
    ----------
    G : nx.Graph
        Social network of herders. Nodes must be integer-indexed 0..N-1.
    h : np.ndarray, shape (N,)
        Private prior for each agent. Positive = leans toward site 1 (+1),
        negative = leans toward site 2 (-1).
    strategies : np.ndarray, shape (N,), dtype bool
        True = Honest, False = Deceptive. Used as the type label evolved by
        Phase C replicator / imitation dynamics.
    p : np.ndarray or None, shape (N,)
        Per-agent honesty probability: agent j reports its true preference with
        probability p[j] and flips the sign with probability 1 - p[j]
        (Eq. 3.15 in the paper). If None, defaults to 1.0 for Honest agents
        and 0.0 for Deceptive agents, recovering the deterministic regime.
    params : dict
        Model parameters. Missing keys fall back to DEFAULT_PARAMS.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        G: nx.Graph,
        h: np.ndarray,
        strategies: np.ndarray,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
        p: np.ndarray | None = None,
    ) -> None:
        self.G = G
        self.N = G.number_of_nodes()
        self.h = np.asarray(h, dtype=float)
        self.strategies = np.asarray(strategies, dtype=bool)  # True = Honest
        self.p = self._resolve_p_vector(p)
        self.rng = np.random.default_rng(seed)

        # Merge with defaults
        raw_params = params or {}
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **raw_params}
        self._resolve_forage_parameters(explicit_abundance="forage_abundance" in raw_params)
        self._validate_params()
        self.alphas = self._resolve_alpha_vector(self.params["alpha"])

        # Spin state: initialised randomly in {-1, +1}
        self.spins = self.rng.choice([-1, 1], size=self.N).astype(float)

        # Pre-compute adjacency list for fast neighbor lookup
        # nodes assumed 0..N-1
        nodes = sorted(G.nodes())
        self._node_index = {n: i for i, n in enumerate(nodes)}
        self._neighbors: list[list[int]] = [
            [self._node_index[nb] for nb in G.neighbors(n)] for n in nodes
        ]

        # Track fraction honest (p) for macro replicator
        n_honest = np.sum(self.strategies)
        self._p: float = float(n_honest) / self.N if self.N > 0 else 0.5

        # History (filled by run())
        self.history: list[dict] = []

    def _validate_params(self) -> None:
        """Validate parameter ranges that affect model semantics."""
        forage_abundance = self.params["forage_abundance"]
        if forage_abundance is not None and float(forage_abundance) <= 0.0:
            raise ValueError(f"forage_abundance must be positive, got {float(forage_abundance)}.")

        alpha_raw = self.params["alpha"]
        alpha_arr = np.asarray(alpha_raw, dtype=float)
        if alpha_arr.ndim == 0:
            if not 0.0 <= float(alpha_arr) <= 1.0:
                raise ValueError(f"alpha must be in [0, 1], got {float(alpha_arr)}.")
        else:
            if len(alpha_arr) != self.N:
                raise ValueError(f"alpha vector must have length {self.N}, got {len(alpha_arr)}.")
            if np.any((alpha_arr < 0.0) | (alpha_arr > 1.0)):
                raise ValueError("all alpha values must be in [0, 1].")

        phase_a_samples = int(self.params["phase_a_samples"])
        if phase_a_samples < 1:
            raise ValueError(f"phase_a_samples must be >= 1, got {phase_a_samples}.")

        phase_a_reset_mode = str(self.params["phase_a_reset_mode"])
        if phase_a_reset_mode not in {"random", "current"}:
            raise ValueError(
                f"phase_a_reset_mode must be 'random' or 'current', got {phase_a_reset_mode!r}."
            )

        macro_assignment_mode = str(self.params["macro_assignment_mode"])
        if macro_assignment_mode not in {"minimal", "resample_all"}:
            raise ValueError(
                "macro_assignment_mode must be 'minimal' or 'resample_all', "
                f"got {macro_assignment_mode!r}."
            )

        eta = float(self.params["eta"])
        if eta < 0.0:
            raise ValueError(f"eta must be non-negative, got {eta}.")

        for key in ("FA", "C"):
            values = self.params[key]
            if len(values) != 2:
                raise ValueError(f"{key} must contain exactly two pasture values, got {values!r}.")

    def _resolve_forage_parameters(
        self,
        *,
        explicit_abundance: bool,
    ) -> None:
        """Resolve symmetric carrying capacities from forage abundance when requested."""
        if not explicit_abundance:
            return

        abundance = float(self.params["forage_abundance"])
        per_site = 0.5 * abundance * float(self.N)
        self.params["FA"] = [per_site, per_site]

    def _quality_gap(self) -> float:
        """Return the quality difference C_A - C_B."""
        C = self.params["C"]
        return float(C[0] - C[1])

    def _resolve_alpha_vector(self, alpha_raw: Any) -> np.ndarray:
        """Broadcast a scalar alpha or validate a per-agent alpha vector."""
        alpha_arr = np.asarray(alpha_raw, dtype=float)
        if alpha_arr.ndim == 0:
            return np.full(self.N, float(alpha_arr), dtype=float)
        return alpha_arr.astype(float, copy=True)

    def _resolve_p_vector(self, p_raw: Any) -> np.ndarray:
        """Resolve per-agent honesty probability, defaulting from strategies."""
        if p_raw is None:
            return self.strategies.astype(float, copy=True)
        p_arr = np.asarray(p_raw, dtype=float)
        if p_arr.ndim == 0:
            p_arr = np.full(self.N, float(p_arr), dtype=float)
        if len(p_arr) != self.N:
            raise ValueError(f"p vector must have length {self.N}, got {len(p_arr)}.")
        if np.any((p_arr < 0.0) | (p_arr > 1.0)):
            raise ValueError("all p values must be in [0, 1].")
        return p_arr.astype(float, copy=True)

    # ------------------------------------------------------------------
    # Phase A: Glauber dynamics
    # ------------------------------------------------------------------

    def _build_jeff(self) -> np.ndarray:
        """
        Build effective coupling vector for each agent: J_ij per neighbor.
        Returns a list of arrays (one per agent) with signed couplings.

        Sign of J_ij is determined by a Bernoulli(p_j) draw: with probability
        p[j] neighbor j reports honestly (J_ij = +J), otherwise it lies
        (J_ij = -J). Draws are quenched within a single Phase A realization.
        """
        J_raw = self.params["J_strength"]
        J_arr = np.asarray(J_raw, dtype=float)
        if J_arr.ndim == 2 and J_arr.shape != (self.N, self.N):
            raise ValueError(
                f"J_strength matrix must have shape ({self.N},{self.N}), got {J_arr.shape}."
            )
        jeff: list[np.ndarray] = []
        for i in range(self.N):
            nbrs = self._neighbors[i]
            if len(nbrs) == 0:
                jeff.append(np.array([], dtype=float))
            else:
                honest_draw = self.rng.random(len(nbrs)) < self.p[nbrs]
                J_i = float(J_arr) if J_arr.ndim == 0 else J_arr[i, nbrs]
                couplings = np.where(honest_draw, J_i, -J_i)
                jeff.append(couplings)
        return jeff  # type: ignore[return-value]

    def _run_phase_a_from_spins(self, initial_spins: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Run Glauber dynamics for n_sweeps sweeps from a provided initial spin state.

        Returns
        -------
        tuple[np.ndarray, float]
            Final spin state and its magnetization.
        """
        beta = self.params["beta"]
        n_sweeps = self.params["n_sweeps"]
        jeff = self._build_jeff()
        quality_gap = self._quality_gap()
        spins = np.asarray(initial_spins, dtype=float).copy()

        for _ in range(n_sweeps):
            # One sweep = N random agent updates
            update_order = self.rng.integers(0, self.N, size=self.N)
            for i in update_order:
                nbrs = self._neighbors[i]
                if len(nbrs) > 0:
                    social_field = float(np.dot(jeff[i], spins[nbrs]))
                else:
                    social_field = 0.0
                alpha_i = self.alphas[i]
                social_weight = 1.0 - alpha_i
                individual_field = alpha_i * quality_gap
                h_eff = self.h[i] + social_weight * social_field + individual_field
                # Glauber probability P(s_i -> +1)
                p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * h_eff))
                spins[i] = 1.0 if self.rng.random() < p_plus else -1.0

        return spins, float(np.mean(spins))

    def run_phase_a(self) -> float:
        """
        Run one Phase A realization in place and return its magnetization.
        """
        self.spins, mag = self._run_phase_a_from_spins(self.spins)
        return mag

    # ------------------------------------------------------------------
    # Phase B: Ecological evaluation
    # ------------------------------------------------------------------

    def _phase_b_for_spins(self, spins: np.ndarray) -> np.ndarray:
        """
        Compute congestion-adjusted payoffs for a provided spin state.

        Returns
        -------
        np.ndarray, shape (N,)
            Per-agent payoffs.
        """
        FA = self.params["FA"]  # [FA1, FA2]
        C = self.params["C"]  # [C1, C2]
        delta = self.params["delta"]
        gamma = self.params["gamma"]
        individual_learning_cost = float(self.params["individual_learning_cost"])

        # Counts at each site
        N1 = float(np.sum(spins == 1.0))
        N2 = float(np.sum(spins == -1.0))

        def site_mu(Nk: float, FAk: float, Ck: float) -> float:
            return Ck / (1.0 + np.exp(gamma * (delta * Nk - FAk)))

        mu1 = site_mu(N1, FA[0], C[0])
        mu2 = site_mu(N2, FA[1], C[1])

        penalties = individual_learning_cost * self.alphas
        payoffs = np.where(spins == 1.0, mu1, mu2) - penalties
        return payoffs

    def run_phase_b(self) -> np.ndarray:
        """
        Compute congestion-adjusted payoffs based on the current spin state.
        """
        return self._phase_b_for_spins(self.spins)

    def run_phase_ab(self) -> tuple[float, np.ndarray]:
        """
        Run Phase A/B, optionally averaging payoffs over multiple micro-realizations.

        When `phase_a_samples > 1`, strategies are held fixed while we average
        over repeated Phase A/B draws. This damps Monte Carlo noise in the macro
        replicator update.
        """
        phase_a_samples = int(self.params["phase_a_samples"])
        if phase_a_samples == 1:
            mag = self.run_phase_a()
            payoffs = self.run_phase_b()
            return mag, payoffs

        reset_mode = str(self.params["phase_a_reset_mode"])
        base_spins = self.spins.copy()
        payoffs_acc = np.zeros(self.N, dtype=float)
        mags: list[float] = []
        last_spins = base_spins

        for _ in range(phase_a_samples):
            if reset_mode == "random":
                init_spins = self.rng.choice([-1.0, 1.0], size=self.N)
            else:
                init_spins = base_spins
            spins_sample, mag_sample = self._run_phase_a_from_spins(init_spins)
            payoffs_acc += self._phase_b_for_spins(spins_sample)
            mags.append(mag_sample)
            last_spins = spins_sample

        self.spins = last_spins.copy()
        return float(np.mean(mags)), payoffs_acc / float(phase_a_samples)

    # ------------------------------------------------------------------
    # Phase C: Strategy update
    # ------------------------------------------------------------------

    def run_phase_c(self, payoffs: np.ndarray) -> None:
        """
        Update strategies in place.

        Parameters
        ----------
        payoffs : np.ndarray, shape (N,)
            Per-agent payoffs from Phase B.
        """
        mode = self.params["update_mode"]
        if mode == "macro":
            self._update_macro(payoffs)
        elif mode in {"micro", "finite"}:
            self._update_micro(payoffs)
        elif mode == "none":
            return
        else:
            raise ValueError(
                f"Unknown update_mode: {mode!r}. Use 'macro', 'micro', 'finite', or 'none'."
            )

    def _update_macro(self, payoffs: np.ndarray) -> None:
        """Mean-field replicator dynamics."""
        strategy_alpha = float(self.params["strategy_alpha"])

        honest_mask = self.strategies
        deceptive_mask = ~self.strategies

        f_H = float(np.mean(payoffs[honest_mask])) if honest_mask.any() else 0.0
        f_D = float(np.mean(payoffs[deceptive_mask])) if deceptive_mask.any() else 0.0

        p = self._p
        phi = p * f_H + (1.0 - p) * f_D
        p_new = p + strategy_alpha * p * (f_H - phi)
        p_new = float(np.clip(p_new, 0.0, 1.0))
        self._p = p_new

        self._assign_macro_strategies(p_new)

    def _assign_macro_strategies(self, p_target: float) -> None:
        """
        Realize the macro replicator state with less stochastic noise.

        `resample_all` reproduces the original iid Bernoulli redraw. The
        default `minimal` mode only flips the minimum number of agents needed
        to match the nearest realized honest count, which dramatically reduces
        count noise and avoids reshuffling the entire network every epoch.
        """
        mode = str(self.params["macro_assignment_mode"])
        if mode == "resample_all":
            self.strategies = self.rng.random(self.N) < p_target
            return

        target_honest = int(np.rint(np.clip(p_target, 0.0, 1.0) * self.N))
        current_honest = int(np.sum(self.strategies))
        if target_honest == current_honest:
            return

        new_strategies = self.strategies.copy()
        if target_honest > current_honest:
            candidates = np.flatnonzero(~new_strategies)
            n_flip = min(target_honest - current_honest, len(candidates))
            if n_flip > 0:
                flip_idx = self.rng.choice(candidates, size=n_flip, replace=False)
                new_strategies[flip_idx] = True
        else:
            candidates = np.flatnonzero(new_strategies)
            n_flip = min(current_honest - target_honest, len(candidates))
            if n_flip > 0:
                flip_idx = self.rng.choice(candidates, size=n_flip, replace=False)
                new_strategies[flip_idx] = False

        self.strategies = new_strategies

    def _update_micro(self, payoffs: np.ndarray) -> None:
        """Finite-population network imitation dynamics on discrete agents."""
        eta = self.params["eta"]
        old_strategies = self.strategies.copy()
        old_alphas = self.alphas.copy()
        old_p = self.p.copy()
        new_strategies = old_strategies.copy()
        new_alphas = old_alphas.copy()
        new_p = old_p.copy()

        for i in range(self.N):
            nbrs = self._neighbors[i]
            if len(nbrs) == 0:
                continue
            j = nbrs[int(self.rng.integers(0, len(nbrs)))]
            if payoffs[j] > payoffs[i]:
                prob = min(1.0, eta * float(payoffs[j] - payoffs[i]))
                if self.rng.random() < prob:
                    new_strategies[i] = old_strategies[j]
                    new_alphas[i] = old_alphas[j]
                    new_p[i] = old_p[j]

        self.strategies = new_strategies
        self.alphas = new_alphas
        self.p = new_p
        self._p = float(np.mean(self.strategies))

    # ------------------------------------------------------------------
    # Epoch and run
    # ------------------------------------------------------------------

    def run_epoch(self) -> dict:
        """
        Execute one full epoch: Phase A -> B -> C.

        Returns
        -------
        dict
            Diagnostics: magnetization, fraction_honest, mean_payoff,
            mean_payoff_honest, mean_payoff_deceptive.
        """
        mag, payoffs = self.run_phase_ab()
        honest_mask = self.strategies.copy()
        deceptive_mask = ~honest_mask

        mean_payoff_H = float(np.mean(payoffs[honest_mask])) if honest_mask.any() else float("nan")
        mean_payoff_D = (
            float(np.mean(payoffs[deceptive_mask])) if deceptive_mask.any() else float("nan")
        )

        self.run_phase_c(payoffs)

        return {
            "magnetization": mag,
            "fraction_honest": float(np.mean(self.strategies)),
            "latent_fraction_honest": float(self._p),
            "mean_alpha": float(np.mean(self.alphas)),
            "std_alpha": float(np.std(self.alphas)),
            "mean_payoff": float(np.mean(payoffs)),
            "mean_payoff_honest": mean_payoff_H,
            "mean_payoff_deceptive": mean_payoff_D,
        }

    def run(self, n_epochs: int) -> list[dict]:
        """
        Run the model for n_epochs epochs.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to simulate.

        Returns
        -------
        list[dict]
            Per-epoch diagnostics (length n_epochs).
        """
        history: list[dict] = []
        for epoch in range(n_epochs):
            diag = self.run_epoch()
            diag["epoch"] = epoch
            history.append(diag)
        self.history = history
        return history
