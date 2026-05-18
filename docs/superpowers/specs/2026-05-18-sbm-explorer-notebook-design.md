# SBM Salutary Duplicity Explorer Notebook — Design

**Date:** 2026-05-18
**Location:** `notebooks/sbm_explorer.ipynb`
**Audience:** Will (researcher) — interactive exploration of the Salutary Duplicity model on a two-block SBM, Colab-compatible.

## Goal

A single notebook that lets us (1) configure a two-block SBM with block-specific prior distributions, (2) inspect the network, priors, and per-site payoff curves, (3) run the Glauber phase-A dynamics forward as an embedded animation with a site-occupancy trace, and (4) live-recompute a payoff-vs-lying-level sweep with adjustable parameters (the `avg_payoff_by_scarcity` figure family).

## Non-goals

- Editing the model itself. All interactivity uses the existing `SalutaryModel` API.
- Reproducing the Quarto figure pipeline. This is a live-exploration notebook, not a paper figure.
- Persisting results. Sweep outputs live only in the notebook session.

## Structure

Four sections, plus a shared constants cell at the top.

### Section 0 — Setup

- On Colab: `pip install` of `networkx matplotlib ipywidgets numpy`; `sys.path` insertion of `../src` so `from salutary_duplicity.model import SalutaryModel` works whether the notebook is opened locally (via the project venv) or uploaded to Colab next to a cloned `src/`.
- `%matplotlib inline`, `import ipywidgets as widgets`.
- Module-level constants: default block sizes, default qualities, default seed.

### Section 1 — Network + priors panel

**Widgets** (`ipywidgets.interactive_output`):

- Block sizes: `N1`, `N2`
- SBM edge probabilities: `p_in`, `p_out`
- Site qualities: `q_A`, `q_B`
- Prior std devs: `sigma_expert`, `sigma_naive`
- Forage params: `F_A`, `F_B`, `gamma`, `delta`
- `seed`

**Prior generation:**

- Block 1 ("experts"): `h_i ~ N(log(q_A/q_B), sigma_expert)`
- Block 2 ("naive"): `h_i ~ N(0, sigma_naive)`

**Output** — three-panel matplotlib figure:

1. Network drawn with `nx.spring_layout(G, seed=...)`. Node color = `h_i` on a diverging colormap. Node shape (`o` vs `s`) distinguishes the two blocks.
2. Histogram of `h_i` split by block, with vertical line at `log(q_A/q_B)`.
3. Per-site payoff curves `μ_k(N_k) = C_k / (1 + exp(γ(δ·N_k - F_k)))` over `N_k ∈ [0, N1+N2]`, with vertical dashed lines at `F_A` and `F_B`.

`interactive_output` is used (not `interact`) so that the three subplots update in one redraw rather than three.

### Section 2 — Run simulation forward (Glauber + animation)

**Inherits** `G`, `h`, `q_A`, `q_B`, `F_A`, `F_B`, `gamma`, `delta` from Section 1.

**New widgets:** `beta`, `J_strength`, `alpha`, `initial_p` (scalar; broadcast to all nodes), `T` (number of phase-A sweeps to animate, capped at 80).

**Procedure:**

```python
model = SalutaryModel(
    G, h,
    strategies=(h > 0),
    p=np.full(N, initial_p),
    params={
        "C": [q_A, q_B], "FA": [F_A, F_B],
        "gamma": gamma, "delta": delta,
        "beta": beta, "J_strength": J_strength, "alpha": alpha,
        "n_sweeps": 1,
        "update_mode": "none",  # no Phase C inside the animation loop
    },
    seed=seed,
)
frames = [model.spins.copy()]
for _ in range(T):
    model.run_phase_a()  # one full sweep over N agents
    frames.append(model.spins.copy())
```

**Outputs:**

- `matplotlib.animation.FuncAnimation` over `frames`, rendered with `.to_jshtml()` (Colab gets a scrub bar / play / pause). Node positions are fixed; node color = spin (+1 = site A, –1 = site B).
- A line plot of `N_A(t)` and `N_B(t)` over the same `T` sweeps, with horizontal dashed lines at `F_A` and `F_B` so over-capacity periods are visually obvious.

Caps: `N ≤ 200`, `T ≤ 80` to keep render under a couple seconds.

### Section 3 — Live `p`-sweep panel

**Always sweeps over `p`.** Other parameters are sliders that re-trigger the recompute.

**Widgets:** `N`, `beta`, `gamma`, `delta`, `q_A`, `q_B`, `F_ratio_sparse`, `F_ratio_abundant`, `strategy_alpha`, `cost`, `alpha`, `n_epochs`, `n_seeds`.

**Snappy defaults** (target ~5–10s per recompute):

- `N=80`, `n_seeds=3`, `n_epochs=30`, `n_sweeps=20` inside phase A.
- 15 p-values, `np.linspace(0, 1, 15)`.
- Two `F_ratio` curves (`F_total = ratio * N`, split equally as `[F_total/2, F_total/2]`): sparse default `0.5`, abundant default `2.0`.

**Per slider change:** Recompute the full grid. For each `(F_ratio, p, seed)`:

```python
model = SalutaryModel(
    G, h, strategies, p=np.full(N, p_val),
    params={..., "FA": [F_total/2, F_total/2], "C": [q_A, q_B], ...},
    seed=seed,
)
for _ in range(n_epochs):
    rec = model.run_epoch()
mean_payoff = rec["mean_payoff"]
```

Network for this section is a *fresh* SBM at each recompute (so `N` is honored), not the network from Section 1.

**Output:**

- One axes, x-axis = `1 - p` (lying level), y-axis = mean per-agent payoff (raw, not delta — the delta version is a Quarto figure, here we want absolute levels too).
- Two curves: "sparse forage" and "plentiful forage", each shaded with ±1 std over seeds.
- Vertical dashed line at the argmax of the sparse curve (the salutary-duplicity peak), if its peak is interior (not at `p=1`).

### Shared constants cell

Defaults live here so the notebook can be calibrated in one place:

```python
DEFAULTS = {
    "N1": 40, "N2": 40,
    "p_in": 0.15, "p_out": 0.02,
    "q_A": 1.5, "q_B": 1.0,
    "sigma_expert": 0.3, "sigma_naive": 0.5,
    "gamma": 2.0, "delta": 1.0,
    "F_A": 40.0, "F_B": 40.0,
    "beta": 1.0, "J_strength": 0.5, "alpha": 0.0,
    "T": 60, "seed": 0,
    "n_p": 15, "n_seeds": 3, "n_epochs": 30,
    "n_sweeps_anim": 1, "n_sweeps_sweep": 20,
}
```

## Performance budget

Phase A scales as `O(N × n_sweeps × <avg degree>)`. At snappy defaults:

- Section 2 animation: `N=80, T=60, n_sweeps=1` → ~5000 spin updates → instantaneous.
- Section 3 sweep: `15 p × 3 seeds × 2 F_ratios × 30 epochs × N=80 × n_sweeps=20` → ~4.3M spin updates. In pure-numpy Python this is ~5–10s. If slower in practice we drop `n_sweeps` to 10 or `n_seeds` to 2.

## Out of scope / explicit cuts

- No JAX. The notebook uses the numpy `SalutaryModel` throughout; JAX gradient work is tracked separately.
- No persistence of sweep results to disk. If the user wants to keep a slice they can copy the array out of the cell.
- No "Run" button — every slider drag recomputes. If interactivity becomes painful at the snappy defaults, the fix is reducing grid size, not adding a button.

## Dependencies

Already in `pyproject.toml`: `networkx`, `numpy`, `matplotlib`, `jupyter`. Adds: `ipywidgets` (Colab has it pre-installed; locally it's an opt-in install — the setup cell does `pip install ipywidgets` defensively).

## Open questions

None blocking. Decisions locked in during brainstorming:

- Block 1 = experts (centered at `log(q_A/q_B)`), Block 2 = naive (centered at 0).
- Section 4 always sweeps `p`; other params are sliders; live recompute on every drag.
- Snappy budget (`N=80`, 3 seeds, 30 epochs, 15 p-values).
- Animation via `FuncAnimation.to_jshtml()`, not a saved GIF file.
