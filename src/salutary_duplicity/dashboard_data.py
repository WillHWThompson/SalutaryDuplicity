from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from salutary_duplicity.sweeps import SweepConfig

REQUIRED_BUNDLE_KEYS = {
    "network_names",
    "alpha_fixed",
    "beta_values",
    "delta_c_values",
    "strategy_alpha_values",
    "cost_values",
    "base_c_values",
    "p_values",
    "reward_mean",
    "reward_std",
    "payoff_gap_mean",
    "payoff_gap_std",
    "final_fraction_mean",
    "final_fraction_std",
    "config_json_by_network",
}


@dataclass(frozen=True)
class SweepDatasetDescriptor:
    dataset_id: str
    path: Path
    relative_path: str
    label: str
    modified_time: float


@dataclass(frozen=True)
class SweepDataset:
    descriptor: SweepDatasetDescriptor
    network_names: np.ndarray
    alpha_fixed: float
    beta_values: np.ndarray
    delta_c_values: np.ndarray
    strategy_alpha_values: np.ndarray
    cost_values: np.ndarray
    base_c_values: np.ndarray
    fa_ratio_values: np.ndarray
    p_values: np.ndarray
    reward_mean: np.ndarray
    reward_std: np.ndarray
    payoff_gap_mean: np.ndarray
    payoff_gap_std: np.ndarray
    final_fraction_mean: np.ndarray
    final_fraction_std: np.ndarray
    network_configs: dict[str, SweepConfig]
    n_seeds: int | None
    created_utc: str | None
    alpha_profile: str | None
    alpha_concentration: float | None
    gap_lim: float


def ensure_fa_axis(array: np.ndarray) -> np.ndarray:
    if array.ndim == 7:
        return np.expand_dims(array, axis=-2)
    return array


def _compact_float(value: float) -> str:
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def _format_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return value
    if stamp.tzinfo is not None and stamp.utcoffset() is not None:
        return stamp.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return stamp.strftime("%Y-%m-%d %H:%M")


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _load_descriptor_metadata(
    path: Path,
) -> tuple[float, str | None, str | None, float | None]:
    with np.load(path, allow_pickle=False) as bundle:
        if not REQUIRED_BUNDLE_KEYS.issubset(bundle.files):
            missing = REQUIRED_BUNDLE_KEYS.difference(bundle.files)
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")

        alpha_fixed = float(np.array(bundle["alpha_fixed"]).item())
        created_utc = None
        if "created_utc" in bundle.files:
            created_utc = str(np.array(bundle["created_utc"]).item())

        alpha_profile = None
        alpha_concentration = None
        config_jsons = np.array(bundle["config_json_by_network"]).astype(str)
        if config_jsons.size:
            try:
                config_payload = json.loads(str(config_jsons[0]))
            except json.JSONDecodeError:
                config_payload = {}
            alpha_profile = config_payload.get("alpha_profile")
            if "alpha_concentration" in config_payload:
                alpha_concentration = float(config_payload["alpha_concentration"])

    return alpha_fixed, created_utc, alpha_profile, alpha_concentration


def _descriptor_label(
    relative_path: str,
    *,
    alpha_fixed: float,
    created_utc: str | None,
    alpha_profile: str | None,
) -> str:
    parts = [relative_path, f"alpha={_compact_float(alpha_fixed)}"]
    if alpha_profile and alpha_profile != "constant":
        parts.append(f"profile={alpha_profile}")
    created_label = _format_timestamp(created_utc)
    if created_label:
        parts.append(created_label)
    return " | ".join(parts)


def discover_sweep_datasets(repo_root: Path) -> list[SweepDatasetDescriptor]:
    sweep_dir = repo_root / "data" / "sweeps"
    candidate_paths: set[Path] = set()
    if sweep_dir.exists():
        candidate_paths.update(sweep_dir.glob("*.npz"))
        workflow_dir = sweep_dir / "workflow"
        if workflow_dir.exists():
            candidate_paths.update(workflow_dir.glob("**/*.npz"))

    descriptors: list[SweepDatasetDescriptor] = []
    for path in sorted(candidate_paths, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            alpha_fixed, created_utc, alpha_profile, alpha_concentration = (
                _load_descriptor_metadata(path)
            )
        except (OSError, ValueError):
            continue
        relative_path = _relative_path(path, repo_root)
        descriptors.append(
            SweepDatasetDescriptor(
                dataset_id=relative_path,
                path=path,
                relative_path=relative_path,
                label=_descriptor_label(
                    relative_path,
                    alpha_fixed=alpha_fixed,
                    created_utc=created_utc,
                    alpha_profile=alpha_profile,
                ),
                modified_time=path.stat().st_mtime,
            )
        )
    return descriptors


def load_sweep_dataset(descriptor: SweepDatasetDescriptor) -> SweepDataset:
    with np.load(descriptor.path, allow_pickle=False) as bundle:
        missing = REQUIRED_BUNDLE_KEYS.difference(bundle.files)
        if missing:
            raise ValueError(f"{descriptor.path} is missing keys: {sorted(missing)}")

        network_names = np.array(bundle["network_names"]).astype(str)
        config_json_by_network = np.array(bundle["config_json_by_network"]).astype(str)
        network_configs = {
            network_name: SweepConfig(**json.loads(config_json))
            for network_name, config_json in zip(network_names, config_json_by_network, strict=True)
        }

        reward_mean = ensure_fa_axis(np.array(bundle["reward_mean"], dtype=float))
        reward_std = ensure_fa_axis(np.array(bundle["reward_std"], dtype=float))
        payoff_gap_mean = ensure_fa_axis(np.array(bundle["payoff_gap_mean"], dtype=float))
        payoff_gap_std = ensure_fa_axis(np.array(bundle["payoff_gap_std"], dtype=float))
        final_fraction_mean = ensure_fa_axis(np.array(bundle["final_fraction_mean"], dtype=float))
        final_fraction_std = ensure_fa_axis(np.array(bundle["final_fraction_std"], dtype=float))

        fa_ratio_values = (
            np.array(bundle["fa_ratio_values"], dtype=float)
            if "fa_ratio_values" in bundle.files
            else np.array([1.0], dtype=float)
        )

        gap_lim = float(np.nanmax(np.abs(payoff_gap_mean)))
        if not np.isfinite(gap_lim) or gap_lim == 0.0:
            gap_lim = 1.0

        created_utc = None
        if "created_utc" in bundle.files:
            created_utc = str(np.array(bundle["created_utc"]).item())

        n_seeds = None
        if "n_seeds" in bundle.files:
            n_seeds = int(np.array(bundle["n_seeds"]).item())

        alpha_profile = None
        alpha_concentration = None
        if config_json_by_network.size:
            config_payload = json.loads(str(config_json_by_network[0]))
            alpha_profile = config_payload.get("alpha_profile")
            if "alpha_concentration" in config_payload:
                alpha_concentration = float(config_payload["alpha_concentration"])

        return SweepDataset(
            descriptor=descriptor,
            network_names=network_names,
            alpha_fixed=float(np.array(bundle["alpha_fixed"]).item()),
            beta_values=np.array(bundle["beta_values"], dtype=float),
            delta_c_values=np.array(bundle["delta_c_values"], dtype=float),
            strategy_alpha_values=np.array(bundle["strategy_alpha_values"], dtype=float),
            cost_values=np.array(bundle["cost_values"], dtype=float),
            base_c_values=np.array(bundle["base_c_values"], dtype=float),
            fa_ratio_values=fa_ratio_values,
            p_values=np.array(bundle["p_values"], dtype=float),
            reward_mean=reward_mean,
            reward_std=reward_std,
            payoff_gap_mean=payoff_gap_mean,
            payoff_gap_std=payoff_gap_std,
            final_fraction_mean=final_fraction_mean,
            final_fraction_std=final_fraction_std,
            network_configs=network_configs,
            n_seeds=n_seeds,
            created_utc=created_utc,
            alpha_profile=alpha_profile,
            alpha_concentration=alpha_concentration,
            gap_lim=gap_lim,
        )


def dataset_choices(descriptors: list[SweepDatasetDescriptor]) -> dict[str, str]:
    return {descriptor.dataset_id: descriptor.label for descriptor in descriptors}
