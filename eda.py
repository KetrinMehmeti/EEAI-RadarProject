"""Exploratory Data Analysis for the UWB multi-person localization dataset.

The script processes one acquisition window at a time, so the complete radar
dataset is never kept in memory. It validates the dataset, computes descriptive
statistics and writes reproducible CSV/JSON summaries and PNG figures.

Example (PowerShell):
    python src/eda.py --data-dir "C:\\path\\to\\dataset" --output-dir outputs/eda

Expected files may be directly inside ``--data-dir`` or in its subdirectories.
Each NPZ file must contain radar_cir_iq, people_xy, people_mask and timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing EDA dependency 'matplotlib'. Install the development dependencies "
        "with: python -m pip install -r src/requirements-eda.txt"
    ) from exc


ROOM_X_METERS = 4.8
ROOM_Y_METERS = 7.2
N_RADARS = 6
N_ANTENNAS = 3
N_RANGE_BINS = 120
MAX_PEOPLE = 4
EXPECTED_KEYS = {"radar_cir_iq", "people_xy", "people_mask", "timestamps"}


@dataclass
class WindowStatistics:
    """Statistics collected for a single acquisition window."""

    window: str
    frames: int
    duration_seconds: float
    sampling_rate_hz: float
    shape_valid: bool
    radar_nan: int
    radar_inf: int
    gt_nan: int
    gt_inf: int
    invalid_mask_values: int
    invalid_coordinates: int
    timestamp_non_increasing: int
    mean_amplitude: float
    std_amplitude: float
    iq_power_ratio: float
    occupancy_0: int
    occupancy_1: int
    occupancy_2: int
    occupancy_3: int
    occupancy_4: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible EDA on the UWB localization dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the dataset NPZ files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eda"),
        help="Directory for CSV, JSON and PNG results (default: outputs/eda).",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Filename pattern used during recursive search (default: *.npz).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional limit for a quick smoke test.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Frames processed together when computing radar statistics.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed used for reproducible sampling."
    )
    return parser.parse_args()


def discover_windows(data_dir: Path, pattern: str, limit: int | None) -> list[Path]:
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    paths = sorted(path for path in data_dir.rglob(pattern) if path.is_file())
    if limit is not None:
        if limit <= 0:
            raise ValueError("--max-windows must be positive")
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(
            f"No files matching {pattern!r} were found under {data_dir}"
        )
    return paths


def safe_rate(timestamps: np.ndarray) -> tuple[float, float, int]:
    """Return duration, median sampling rate, and non-increasing step count."""
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 2 or not np.all(np.isfinite(ts)):
        return 0.0, math.nan, 0
    differences = np.diff(ts)
    invalid_steps = int(np.count_nonzero(differences <= 0))
    positive = differences[differences > 0]
    duration = float(ts[-1] - ts[0])
    rate = float(1.0 / np.median(positive)) if positive.size else math.nan
    return duration, rate, invalid_steps


def iter_chunks(array: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, array.shape[0], chunk_size):
        yield array[start : start + chunk_size]


def analyse_window(
    path: Path,
    chunk_size: int,
    range_sum: np.ndarray,
    radar_sum: np.ndarray,
) -> tuple[WindowStatistics, np.ndarray, np.ndarray, np.ndarray]:
    """Analyse one NPZ and update global amplitude accumulators."""
    with np.load(path, allow_pickle=False) as data:
        missing = EXPECTED_KEYS.difference(data.files)
        if missing:
            raise KeyError(f"{path.name}: missing arrays {sorted(missing)}")

        radar = data["radar_cir_iq"]
        xy = np.asarray(data["people_xy"])
        mask_raw = np.asarray(data["people_mask"])
        timestamps = np.asarray(data["timestamps"])

        frames = int(radar.shape[0]) if radar.ndim else 0
        expected_radar = (frames, N_RADARS, N_ANTENNAS, N_RANGE_BINS, 2)
        shape_valid = (
            radar.shape == expected_radar
            and xy.shape == (frames, MAX_PEOPLE, 2)
            and mask_raw.shape == (frames, MAX_PEOPLE)
            and timestamps.shape == (frames,)
        )
        if not shape_valid:
            raise ValueError(
                f"{path.name}: unexpected shapes: radar={radar.shape}, "
                f"xy={xy.shape}, mask={mask_raw.shape}, timestamps={timestamps.shape}"
            )

        invalid_mask_values = int(np.count_nonzero(~np.isin(mask_raw, [0, 1, False, True])))
        mask = mask_raw.astype(bool)
        valid_xy = xy[mask]
        gt_nan = int(np.isnan(valid_xy).sum())
        gt_inf = int(np.isinf(valid_xy).sum())
        if valid_xy.size:
            invalid_coordinates = int(
                np.count_nonzero(
                    (valid_xy[:, 0] < 0)
                    | (valid_xy[:, 0] > ROOM_X_METERS)
                    | (valid_xy[:, 1] < 0)
                    | (valid_xy[:, 1] > ROOM_Y_METERS)
                )
            )
        else:
            invalid_coordinates = 0

        occupancy = mask.sum(axis=1).astype(np.int64)
        occupancy_counts = np.bincount(occupancy, minlength=MAX_PEOPLE + 1)
        duration, sampling_rate, timestamp_errors = safe_rate(timestamps)

        radar_nan = radar_inf = element_count = 0
        amplitude_sum = amplitude_sq_sum = i_power = q_power = 0.0
        local_range_sum = np.zeros(N_RANGE_BINS, dtype=np.float64)
        local_radar_sum = np.zeros(N_RADARS, dtype=np.float64)

        for chunk in iter_chunks(radar, chunk_size):
            radar_nan += int(np.isnan(chunk).sum())
            radar_inf += int(np.isinf(chunk).sum())
            # Non-finite values are excluded from descriptive signal statistics.
            i = np.nan_to_num(chunk[..., 0], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
            q = np.nan_to_num(chunk[..., 1], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
            amplitude = np.hypot(i, q)
            amplitude_sum += float(amplitude.sum())
            amplitude_sq_sum += float(np.square(amplitude).sum())
            element_count += int(amplitude.size)
            i_power += float(np.square(i).sum())
            q_power += float(np.square(q).sum())
            local_range_sum += amplitude.sum(axis=(0, 1, 2))
            local_radar_sum += amplitude.sum(axis=(0, 2, 3))

        mean_amplitude = amplitude_sum / max(element_count, 1)
        variance = max(amplitude_sq_sum / max(element_count, 1) - mean_amplitude**2, 0.0)
        std_amplitude = math.sqrt(variance)
        iq_ratio = i_power / q_power if q_power > 0 else math.nan
        range_sum += local_range_sum
        radar_sum += local_radar_sum

        stats = WindowStatistics(
            window=path.name,
            frames=frames,
            duration_seconds=duration,
            sampling_rate_hz=sampling_rate,
            shape_valid=shape_valid,
            radar_nan=radar_nan,
            radar_inf=radar_inf,
            gt_nan=gt_nan,
            gt_inf=gt_inf,
            invalid_mask_values=invalid_mask_values,
            invalid_coordinates=invalid_coordinates,
            timestamp_non_increasing=timestamp_errors,
            mean_amplitude=mean_amplitude,
            std_amplitude=std_amplitude,
            iq_power_ratio=iq_ratio,
            occupancy_0=int(occupancy_counts[0]),
            occupancy_1=int(occupancy_counts[1]),
            occupancy_2=int(occupancy_counts[2]),
            occupancy_3=int(occupancy_counts[3]),
            occupancy_4=int(occupancy_counts[4]),
        )
        return stats, occupancy, valid_xy.copy(), mask.copy()


def save_csv(rows: list[WindowStatistics], output_path: Path) -> None:
    records = [asdict(row) for row in rows]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    rows: list[WindowStatistics],
    occupancy_counts: np.ndarray,
    coordinates: np.ndarray,
    range_profile: np.ndarray,
    radar_profile: np.ndarray,
    output_dir: Path,
) -> None:
    # Occupancy distribution.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(np.arange(5), occupancy_counts, color="#3478bf")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set(title="Occupancy distribution", xlabel="People in frame", ylabel="Frames")
    ax.set_xticks(np.arange(5))
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir / "occupancy_distribution.png")

    # Spatial ground-truth density; hexbin remains readable with many points.
    fig, ax = plt.subplots(figsize=(6, 7))
    if coordinates.size:
        density = ax.hexbin(
            coordinates[:, 0], coordinates[:, 1], gridsize=35, cmap="viridis", mincnt=1
        )
        fig.colorbar(density, ax=ax, label="Ground-truth observations")
    ax.set(
        title="Ground-truth spatial coverage",
        xlabel="x [m]",
        ylabel="y [m]",
        xlim=(0, ROOM_X_METERS),
        ylim=(0, ROOM_Y_METERS),
        aspect="equal",
    )
    ax.grid(alpha=0.2)
    save_figure(fig, output_dir / "spatial_coverage.png")

    # Mean amplitude along range bins.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(N_RANGE_BINS), range_profile, color="#ca5c28")
    ax.set(title="Mean CIR amplitude by range bin", xlabel="Range bin", ylabel="Mean amplitude")
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir / "range_bin_profile.png")

    # Mean amplitude per radar.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(np.arange(N_RADARS), radar_profile, color="#4d9b73")
    ax.bar_label(bars, fmt="%.3g", padding=3)
    ax.set(title="Mean CIR amplitude by radar", xlabel="Radar index", ylabel="Mean amplitude")
    ax.set_xticks(np.arange(N_RADARS))
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir / "radar_amplitude.png")

    # Per-window signal level, useful for detecting acquisition shifts.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    names = [row.window.replace("window_", "").replace(".npz", "") for row in rows]
    means = [row.mean_amplitude for row in rows]
    ax.plot(names, means, marker="o", linewidth=1.5)
    ax.set(title="Mean CIR amplitude by acquisition window", xlabel="Window", ylabel="Mean amplitude")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir / "window_amplitude.png")


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    windows = discover_windows(args.data_dir, args.pattern, args.max_windows)
    print(f"Found {len(windows)} acquisition window(s).")

    rows: list[WindowStatistics] = []
    global_occupancy = np.zeros(MAX_PEOPLE + 1, dtype=np.int64)
    coordinates_parts: list[np.ndarray] = []
    range_sum = np.zeros(N_RANGE_BINS, dtype=np.float64)
    radar_sum = np.zeros(N_RADARS, dtype=np.float64)
    total_frames = 0

    for index, path in enumerate(windows, start=1):
        print(f"[{index:02d}/{len(windows):02d}] Analysing {path.name}")
        row, occupancy, valid_xy, _ = analyse_window(
            path, args.chunk_size, range_sum, radar_sum
        )
        rows.append(row)
        global_occupancy += np.bincount(occupancy, minlength=MAX_PEOPLE + 1)
        coordinates_parts.append(valid_xy)
        total_frames += row.frames

    coordinates = (
        np.concatenate(coordinates_parts, axis=0)
        if coordinates_parts
        else np.empty((0, 2), dtype=np.float32)
    )
    # Each frame contributes R*A values to every bin and A*B values to every radar.
    range_denominator = max(total_frames * N_RADARS * N_ANTENNAS, 1)
    radar_denominator = max(total_frames * N_ANTENNAS * N_RANGE_BINS, 1)
    range_profile = range_sum / range_denominator
    radar_profile = radar_sum / radar_denominator

    save_csv(rows, args.output_dir / "window_statistics.csv")
    make_plots(
        rows,
        global_occupancy,
        coordinates,
        range_profile,
        radar_profile,
        args.output_dir,
    )

    summary = {
        "data_directory": str(args.data_dir.resolve()),
        "windows": len(rows),
        "frames": total_frames,
        "ground_truth_people_observations": int(coordinates.shape[0]),
        "occupancy_frames": {str(i): int(value) for i, value in enumerate(global_occupancy)},
        "data_quality": {
            "radar_nan": sum(row.radar_nan for row in rows),
            "radar_inf": sum(row.radar_inf for row in rows),
            "ground_truth_nan": sum(row.gt_nan for row in rows),
            "ground_truth_inf": sum(row.gt_inf for row in rows),
            "invalid_mask_values": sum(row.invalid_mask_values for row in rows),
            "out_of_bounds_coordinates": sum(row.invalid_coordinates for row in rows),
            "non_increasing_timestamp_steps": sum(
                row.timestamp_non_increasing for row in rows
            ),
        },
        "signal": {
            "mean_amplitude_by_radar": radar_profile.tolist(),
            "mean_amplitude_by_range_bin": range_profile.tolist(),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    print(f"EDA completed. Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
