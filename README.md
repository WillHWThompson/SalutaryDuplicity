# SalutaryDuplicity

Research code for the SalutaryDuplicity learning and imitation model.

The repository has three main uses:

1. Run parameter sweeps over the model and save results as `.npz` bundles.
2. Explore saved sweep bundles in the Shiny dashboard.
3. Reproduce figures and exploratory analysis from the Quarto notebooks.

## Repository Layout

- `dashboard/`: standalone Shiny for Python dashboard.
- `src/salutary_duplicity/`: model, sweep, plotting-style, and dataset-loading code.
- `scripts/`: direct entry points for running sweeps and rendering notebooks.
- `configs/`: Snakemake config files for named experiment runs.
- `workflow/`: Snakemake workflow helpers and rules.
- `data/sweeps/`: expected location for standalone sweep outputs.
- `data/sweeps/workflow/`: expected location for workflow-managed outputs.
- `notebooks/`: Quarto notebooks and rendered HTML outputs.
- `test/`: pytest suite.

## Environment Setup

This project uses `uv` for environment management and execution.

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Quarto, if you want to preview or render the notebooks

### Install dependencies

From the repository root:

```bash
uv sync
```

That creates or updates `.venv/` and installs the package plus development dependencies from `pyproject.toml`.

## Quick Start

If you mainly want to see the dashboard:

```bash
uv sync
uv run python scripts/run_multiparam_network_sweep.py --workers 4
uv run shiny run --reload dashboard/app.py
```

Then open the local URL printed by Shiny, typically `http://127.0.0.1:8000`.

Important: the dashboard requires at least one compatible `.npz` dataset under `data/sweeps/` or `data/sweeps/workflow/`. If no dataset exists, the app raises a startup error and tells you to generate one first.

## Running the Dashboard

The dashboard lives at [dashboard/app.py](dashboard/app.py) and auto-discovers compatible sweep bundles at startup.

### Start the dashboard

```bash
uv run shiny run --reload dashboard/app.py
```

### What the dashboard expects

The app scans:

- `data/sweeps/*.npz`
- `data/sweeps/workflow/**/*.npz`

Each dataset selector entry is built from bundle metadata such as:

- relative file path
- fixed decision-level `alpha`
- optional alpha profile
- creation time, when available

### Core dashboard concepts

The dashboard distinguishes between two different uses of alpha:

- `alpha` / `alpha_fixed`: the decision-level social-versus-individual learning mix used inside Phase A/B.
- `strategy_alpha`: the replicator learning rate used in Phase C.

The main controls let you switch:

- dataset bundle
- surface metric
- network family
- the parameter on the heatmap y-axis
- fixed indices for the remaining sweep dimensions
- number of trajectory seeds shown in the lower panels

### Common dashboard workflow

1. Generate one or more sweep bundles.
2. Launch the dashboard.
3. Pick a dataset from the sidebar.
4. Choose the metric and varying parameter.
5. Compare slices, network differences, and trajectory behavior.

## Generating Sweep Data Directly

The simplest way to produce a dashboard-compatible dataset is to run one of the Python scripts in `scripts/`.

### Multi-parameter sweep

Primary script: [scripts/run_multiparam_network_sweep.py](scripts/run_multiparam_network_sweep.py)

Default example:

```bash
uv run python scripts/run_multiparam_network_sweep.py
```

This writes a bundle by default to:

```text
data/sweeps/multiparam_network_sweep_alpha_fixed.npz
```

Example with explicit output path and smaller run:

```bash
uv run python scripts/run_multiparam_network_sweep.py \
  --beta-values 0.5,2.0 \
  --delta-c-values 0.0,1.0 \
  --strategy-alpha-values 0.0,0.05 \
  --cost-values 0.0,1.0 \
  --base-c-values 1.0 \
  --fa-ratio-values 0.5,1.0,2.0 \
  --p-points 9 \
  --n-seeds 2 \
  --workers 4 \
  --out data/sweeps/my_small_run.npz
```

Important CLI flags:

- `--networks`: comma-separated network names, currently `karate,sbm` by default.
- `--alpha-fixed`: fixed mean decision alpha for Phase A/B.
- `--alpha-profile`: `constant` or `beta`.
- `--alpha-concentration`: concentration parameter when `alpha-profile=beta`.
- `--beta-values`
- `--delta-c-values`
- `--strategy-alpha-values`
- `--cost-values`
- `--base-c-values`
- `--fa-ratio-values`
- `--p-points`
- `--n-seeds`
- `--workers`
- `--n-epochs`
- `--burn-in`
- `--n-sweeps`
- `--update-mode`
- `--out`

### Network alpha/p comparison sweep

Secondary script: [scripts/run_network_alpha_p_comparison.py](scripts/run_network_alpha_p_comparison.py)

This script is more specialized. It compares reward landscapes across network families over an `(alpha, p0)` grid and is useful for isolating Phase A/B effects or studying low-noise replicator settings.

Example:

```bash
uv run python scripts/run_network_alpha_p_comparison.py \
  --delta-c 1.0 \
  --strategy-alpha 0.05 \
  --update-mode macro \
  --n-seeds 3 \
  --workers 4 \
  --out data/sweeps/network_alpha_compare.npz
```

## Running Sweeps with Snakemake

The repository also has a config-driven Snakemake workflow for managing repeatable batches of runs.

Main workflow files:

- [Snakefile](Snakefile)
- [workflow/rules/sweeps.smk](workflow/rules/sweeps.smk)
- [workflow/lib/experiment_plan.py](workflow/lib/experiment_plan.py)

### Default workflow run

```bash
uv run snakemake -j 4 sweeps
```

That loads `configs/default_run.yaml` and builds all outputs listed by the active config.

### Run a named config

```bash
uv run snakemake -j 4 sweeps --configfile configs/replicator_low_noise.yaml
```

Other existing configs:

- `configs/default_run.yaml`
- `configs/per_agent_alpha.yaml`
- `configs/replicator_low_noise.yaml`
- `configs/sweep_template.yaml`

### Dry-run the workflow

Use this before launching a larger batch:

```bash
uv run snakemake -n sweeps --configfile configs/default_run.yaml
```

### Force reruns

If you want to rerun targets whose output files already exist:

```bash
uv run snakemake -j 4 -F sweeps --configfile configs/default_run.yaml
```

## How the Snakemake Configs Work

Each config file has the same high-level structure:

```yaml
name: "example_run"
experiment: "multiparam_sweep"
script: "scripts/run_multiparam_network_sweep.py"
output_dir: "data/sweeps/workflow/example_run"

params:
  alpha_fixed: 0.2
  beta_values: "0.5,1.0,2.0"
  ...

sweep:
  strategy_alpha: [0.05, 0.2]
  phase_a_samples: [8, 16]
```

The fields mean:

- `name`: label for the run family.
- `experiment`: logical experiment name; currently mapped to a default script if `script` is omitted.
- `script`: Python entry point to execute for each row.
- `output_dir`: directory where `.npz` outputs are written.
- `params`: base CLI arguments passed to every run.
- `sweep`: optional cartesian product of parameter overrides.

For each row in the cartesian product, the workflow:

1. Merges `params` with that row's overrides.
2. Builds a stable `instance_id`.
3. Runs the chosen script with `uv run python ...`.
4. Writes one `.npz` file to `output_dir/{instance_id}.npz`.

### Example: create a new workflow config

Copy the template:

```bash
cp configs/sweep_template.yaml configs/my_run.yaml
```

Then edit:

- `name`
- `output_dir`
- `params`
- optional `sweep`

Run it with:

```bash
uv run snakemake -j 4 sweeps --configfile configs/my_run.yaml
```

## Existing Config Files

### `configs/default_run.yaml`

Baseline multi-parameter workflow run over:

- networks `karate,sbm`
- beta values `0.5,1.0,2.0`
- Delta C values `0.0,0.5,1.0`
- strategy alpha values `0.0,0.05`
- cost values `0.0,1.0`
- base c values `0.75,1.0,1.25`
- forage abundance ratios `0.5,1.0,2.0`

### `configs/per_agent_alpha.yaml`

Like the baseline run, but with heterogeneous per-agent alpha draws via:

- `alpha_profile: "beta"`
- `alpha_concentration: 10.0`
- `phase_a_samples: 4`

### `configs/replicator_low_noise.yaml`

Uses the `network_alpha_p_comparison` experiment and sweeps over:

- `strategy_alpha: [0.05, 0.2]`
- `phase_a_samples: [8, 16]`

It is aimed at lower-noise replicator comparisons with longer runs.

## Output Files

### Standalone script outputs

Direct script runs usually write to:

- `data/sweeps/*.npz`

### Workflow outputs

Snakemake runs usually write to:

- `data/sweeps/workflow/<config-name>/*.npz`

These files are the primary input format for the dashboard.

## Notebooks and Reports

Quarto notebooks live in `notebooks/`.

Preview a notebook:

```bash
./scripts/preview_dashboard.sh notebooks/3-23-26.qmd
```

Render a notebook:

```bash
./scripts/render_dashboard.sh notebooks/3-23-26.qmd
```

These helper scripts set:

- `QUARTO_PYTHON` to the repo virtualenv Python
- `MPLCONFIGDIR` to a writable temporary directory

Despite their names, these scripts are for Quarto notebook preview/render, not for launching the Shiny dashboard.

## Tests and Checks

Run the test suite:

```bash
uv run pytest
```

Run Ruff:

```bash
uv run ruff check .
```

## Typical Workflows

### Explore the dashboard with one freshly generated bundle

```bash
uv sync
uv run python scripts/run_multiparam_network_sweep.py --workers 4
uv run shiny run --reload dashboard/app.py
```

### Run a workflow-managed experiment and inspect it in the dashboard

```bash
uv sync
uv run snakemake -j 4 sweeps --configfile configs/per_agent_alpha.yaml
uv run shiny run --reload dashboard/app.py
```

### Prototype a new sweep config

```bash
cp configs/sweep_template.yaml configs/my_run.yaml
uv run snakemake -n sweeps --configfile configs/my_run.yaml
uv run snakemake -j 4 sweeps --configfile configs/my_run.yaml
```

## Troubleshooting

### The dashboard says no dataset was found

Generate a bundle first with either:

```bash
uv run python scripts/run_multiparam_network_sweep.py
```

or:

```bash
uv run snakemake -j 4 sweeps
```

### The dashboard starts but does not show the dataset I expect

Check that the output file:

- is a `.npz`
- lives under `data/sweeps/` or `data/sweeps/workflow/`
- has the expected bundle keys written by the sweep scripts

### Snakemake does nothing

That usually means the target outputs already exist and are considered up to date. Try:

```bash
uv run snakemake -n sweeps --configfile configs/default_run.yaml
uv run snakemake -j 4 -F sweeps --configfile configs/default_run.yaml
```

### Quarto preview or render fails

Make sure Quarto is installed separately; it is not installed through `uv sync`.
