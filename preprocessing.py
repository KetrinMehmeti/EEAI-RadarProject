"""Shared preprocessing for training and final UWB radar inference.

All operations in this module use NumPy and can be reproduced in
``submission/code.py``. Training-only statistics are supplied explicitly; no
validation or test samples are used to estimate normalisation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ROOM_X_METERS = 4.8
ROOM_Y_METERS = 7.2
MAX_PEOPLE = 4


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration shared by preprocessing, training, and inference."""

    radar_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    antenna_mode: str = "none"
    range_bin_start: int = 0
    range_bin_stop: int = 120
    log_compression: bool = True
    background_mode: str = "none"
    ema_alpha: float = 0.98
    epsilon: float = 1e-6

    def validate(self) -> None:
        if not self.radar_indices:
            raise ValueError("At least one radar must be selected")
        if len(set(self.radar_indices)) != len(self.radar_indices):
            raise ValueError("radar_indices must not contain duplicates")
        if any(index < 0 or index >= 6 for index in self.radar_indices):
            raise ValueError("Radar indices must be between 0 and 5")
        if self.antenna_mode not in {"none", "mean", "max"}:
            raise ValueError("antenna_mode must be 'none', 'mean', or 'max'")
        if not 0 <= self.range_bin_start < self.range_bin_stop <= 120:
            raise ValueError("Invalid range-bin interval")
        if self.background_mode not in {"none", "ema"}:
            raise ValueError("background_mode must be 'none' or 'ema'")
        if not 0.0 < self.ema_alpha < 1.0:
            raise ValueError("ema_alpha must lie strictly between 0 and 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def iq_to_amplitude(radar_cir_iq: np.ndarray) -> np.ndarray:
    """Convert I/Q samples ``(..., 2)`` to float32 magnitude."""
    array = np.asarray(radar_cir_iq)
    if array.ndim != 5 or array.shape[-1] != 2:
        raise ValueError(f"Expected radar shape (T, 6, 3, 120, 2), received {array.shape}")
    i = array[..., 0].astype(np.float32)
    q = array[..., 1].astype(np.float32)
    return np.hypot(i, q, dtype=np.float32)


def select_and_reduce(amplitude: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Select sensors/bins and optionally aggregate the antenna dimension."""
    config.validate()
    features = amplitude[
        :, list(config.radar_indices), :, config.range_bin_start : config.range_bin_stop
    ]
    if config.antenna_mode == "mean":
        features = features.mean(axis=2, dtype=np.float32)
    elif config.antenna_mode == "max":
        features = features.max(axis=2)
    return features.astype(np.float32, copy=False)


def apply_log_compression(features: np.ndarray, enabled: bool) -> np.ndarray:
    """Compress the large positive CIR dynamic range using log(1 + amplitude)."""
    if not enabled:
        return features.astype(np.float32, copy=False)
    return np.log1p(features, dtype=np.float32)


def causal_ema_residual(features: np.ndarray, alpha: float) -> np.ndarray:
    """Subtract a causal EMA background independently for each acquisition.

    The first frame initialises the background and consequently has a zero
    residual. This function is optional because long stationary intervals can
    be absorbed by the EMA, which may be harmful for seated subjects.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if len(features) == 0:
        return features.astype(np.float32, copy=True)
    output = np.empty_like(features, dtype=np.float32)
    background = features[0].astype(np.float32, copy=True)
    output[0] = 0.0
    for frame_index in range(1, len(features)):
        current = features[frame_index]
        output[frame_index] = current - background
        background = alpha * background + (1.0 - alpha) * current
    return output


def transform_signal(radar_cir_iq: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply deterministic signal transforms before normalisation."""
    features = iq_to_amplitude(radar_cir_iq)
    features = select_and_reduce(features, config)
    features = apply_log_compression(features, config.log_compression)
    if config.background_mode == "ema":
        features = causal_ema_residual(features, config.ema_alpha)
    if not np.isfinite(features).all():
        raise ValueError("Signal preprocessing produced non-finite values")
    return features


def transform_signed_iq_channels(
    radar_cir_iq: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Preserve signed I/Q information as two channels per antenna.

    Output shape is ``(T, selected_radars, 6, selected_bins)`` when all three
    antennas are retained. Signed log compression reduces dynamic range without
    discarding the sign/phase cues lost by magnitude-only preprocessing.
    """
    config.validate()
    if config.antenna_mode != "none":
        raise ValueError("Signed I/Q channels require antenna_mode='none'")
    array = np.asarray(radar_cir_iq)
    if array.ndim != 5 or array.shape[-1] != 2:
        raise ValueError(f"Expected radar shape (T, 6, 3, 120, 2), received {array.shape}")
    selected = array[
        :, list(config.radar_indices), :, config.range_bin_start : config.range_bin_stop, :
    ].astype(np.float32)
    if config.log_compression:
        selected = np.sign(selected) * np.log1p(np.abs(selected), dtype=np.float32)
    # (T,R,A,B,IQ) -> (T,R,A,IQ,B) -> (T,R,A*IQ,B)
    selected = np.transpose(selected, (0, 1, 2, 4, 3))
    features = selected.reshape(
        selected.shape[0], selected.shape[1], selected.shape[2] * 2, selected.shape[4]
    )
    if not np.isfinite(features).all():
        raise ValueError("Signed I/Q preprocessing produced non-finite values")
    return features.astype(np.float32, copy=False)


def transform_fused_magnitude_iq(
    radar_cir_iq: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Fuse magnitude and signed I/Q as nine channels per radar.

    Channel order is ``[A0-mag, A1-mag, A2-mag, A0-I, A0-Q, A1-I,
    A1-Q, A2-I, A2-Q]``. Both branches use the same signed/logarithmic
    compression policy encoded by ``config``. This representation combines
    the robust energy cue used by V2 with the quadrature cues tested by V5.
    """
    if config.antenna_mode != "none":
        raise ValueError("Fused magnitude/IQ channels require antenna_mode='none'")
    magnitude = transform_signal(radar_cir_iq, config)
    signed_iq = transform_signed_iq_channels(radar_cir_iq, config)
    features = np.concatenate((magnitude, signed_iq), axis=2)
    if not np.isfinite(features).all():
        raise ValueError("Fused magnitude/IQ preprocessing produced non-finite values")
    return features.astype(np.float32, copy=False)


def inter_antenna_phase_features(
    radar_cir_iq: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Return stable sine/cosine phase differences for three antenna pairs.

    Direct angle subtraction is discontinuous at ``-pi/pi``. The normalised
    complex cross-product provides the equivalent ``cos(delta_phi)`` and
    ``sin(delta_phi)`` values without an ``atan2`` operation. Channel order is
    ``[cos01, sin01, cos12, sin12, cos02, sin02]``.
    """
    config.validate()
    if config.antenna_mode != "none":
        raise ValueError("Phase-difference features require antenna_mode='none'")
    array = np.asarray(radar_cir_iq)
    if array.ndim != 5 or array.shape[-1] != 2 or array.shape[2] != 3:
        raise ValueError(f"Expected radar shape (T,6,3,120,2), received {array.shape}")
    selected = array[
        :, list(config.radar_indices), :, config.range_bin_start : config.range_bin_stop, :
    ].astype(np.float32)
    i_values, q_values = selected[..., 0], selected[..., 1]
    magnitude = np.sqrt(i_values * i_values + q_values * q_values, dtype=np.float32)
    pair_channels = []
    for first, second in ((0, 1), (1, 2), (0, 2)):
        denominator = magnitude[:, :, first] * magnitude[:, :, second] + config.epsilon
        cosine = (
            i_values[:, :, first] * i_values[:, :, second]
            + q_values[:, :, first] * q_values[:, :, second]
        ) / denominator
        sine = (
            q_values[:, :, first] * i_values[:, :, second]
            - i_values[:, :, first] * q_values[:, :, second]
        ) / denominator
        pair_channels.extend((cosine, sine))
    features = np.stack(pair_channels, axis=2)
    features = np.clip(features, -1.0, 1.0)
    if not np.isfinite(features).all():
        raise ValueError("Phase-difference preprocessing produced non-finite values")
    return features.astype(np.float32, copy=False)


def causal_ema_residual_stateful(
    features: np.ndarray,
    alpha: float,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply causal EMA residuals while preserving state across chunks."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    features = np.asarray(features, dtype=np.float32)
    if len(features) == 0:
        if background is None:
            raise ValueError("Cannot initialise EMA state from an empty block")
        return features.copy(), np.asarray(background, dtype=np.float32)
    output = np.empty_like(features, dtype=np.float32)
    start = 0
    if background is None:
        background = features[0].astype(np.float32, copy=True)
        output[0] = 0.0
        start = 1
    else:
        background = np.asarray(background, dtype=np.float32).copy()
        if background.shape != features.shape[1:]:
            raise ValueError("EMA background shape does not match feature shape")
    for frame_index in range(start, len(features)):
        current = features[frame_index]
        output[frame_index] = current - background
        background = alpha * background + (1.0 - alpha) * current
    return output, background.astype(np.float32, copy=False)


def transform_phase_ema_temporal(
    radar_cir_iq: np.ndarray,
    config: PreprocessingConfig,
    ema_background: np.ndarray | None = None,
    previous_base_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal current/previous features with magnitude, EMA, I/Q, phase.

    The 18 base channels are ``3 magnitude + 3 EMA magnitude residual +
    6 signed I/Q + 6 phase-difference`` channels. Concatenating the current
    and previous base features produces 36 channels per radar. State is
    returned explicitly so callers can preserve continuity across chunks and
    reset it only at acquisition-window boundaries.
    """
    config.validate()
    amplitude = iq_to_amplitude(radar_cir_iq)
    magnitude = select_and_reduce(amplitude, config)
    magnitude = apply_log_compression(magnitude, config.log_compression)
    ema_residual, ema_background = causal_ema_residual_stateful(
        magnitude, config.ema_alpha, ema_background
    )
    signed_iq = transform_signed_iq_channels(radar_cir_iq, config)
    phase = inter_antenna_phase_features(radar_cir_iq, config)
    base_features = np.concatenate((magnitude, ema_residual, signed_iq, phase), axis=2)
    previous = np.empty_like(base_features)
    if previous_base_features is None:
        previous[0] = 0.0
    else:
        previous_state = np.asarray(previous_base_features, dtype=np.float32)
        if previous_state.shape != base_features.shape[1:]:
            raise ValueError("Previous temporal feature state has an invalid shape")
        previous[0] = previous_state
    if len(base_features) > 1:
        previous[1:] = base_features[:-1]
    temporal_features = np.concatenate((base_features, previous), axis=2)
    if not np.isfinite(temporal_features).all():
        raise ValueError("Phase/EMA temporal preprocessing produced non-finite values")
    return (
        temporal_features.astype(np.float32, copy=False),
        ema_background,
        base_features[-1].astype(np.float32, copy=True),
    )


def clip_valid_coordinates(people_xy: np.ndarray, people_mask: np.ndarray) -> np.ndarray:
    """Clip valid labels to the official room bounds without changing masks."""
    coordinates = np.asarray(people_xy, dtype=np.float32).copy()
    mask = np.asarray(people_mask).astype(bool)
    if coordinates.shape[:-1] != mask.shape or coordinates.shape[-1] != 2:
        raise ValueError("people_xy and people_mask have incompatible shapes")
    coordinates[..., 0][mask] = np.clip(
        coordinates[..., 0][mask], 0.0, ROOM_X_METERS
    )
    coordinates[..., 1][mask] = np.clip(
        coordinates[..., 1][mask], 0.0, ROOM_Y_METERS
    )
    return coordinates


def build_sorted_targets(people_xy: np.ndarray, people_mask: np.ndarray) -> np.ndarray:
    """Create targets ``(T, 4, 3)`` as ``[confidence, x_norm, y_norm]``.

    Valid people are sorted lexicographically by x and then y. Sorting provides
    a deterministic baseline assignment while retaining all presence labels.
    A permutation-invariant objective can be evaluated later if crossings make
    this assignment unstable.
    """
    coordinates = clip_valid_coordinates(people_xy, people_mask)
    mask = np.asarray(people_mask).astype(bool)
    targets = np.zeros((len(coordinates), MAX_PEOPLE, 3), dtype=np.float32)
    for frame_index in range(len(coordinates)):
        valid_xy = coordinates[frame_index, mask[frame_index]]
        if not len(valid_xy):
            continue
        order = np.lexsort((valid_xy[:, 1], valid_xy[:, 0]))
        sorted_xy = valid_xy[order]
        count = len(sorted_xy)
        targets[frame_index, :count, 0] = 1.0
        targets[frame_index, :count, 1] = sorted_xy[:, 0] / ROOM_X_METERS
        targets[frame_index, :count, 2] = sorted_xy[:, 1] / ROOM_Y_METERS
    return targets


def per_radar_moments(feature_blocks: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute leakage-free per-radar mean/std from training feature blocks."""
    total_sum = None
    total_square_sum = None
    total_values_per_radar = 0
    for features in feature_blocks:
        if features.ndim not in {3, 4}:
            raise ValueError(f"Expected (T,R,B) or (T,R,A,B), received {features.shape}")
        reduction_axes = tuple(axis for axis in range(features.ndim) if axis != 1)
        block_sum = features.sum(axis=reduction_axes, dtype=np.float64)
        block_square_sum = np.square(features, dtype=np.float64).sum(
            axis=reduction_axes, dtype=np.float64
        )
        values_per_radar = int(np.prod([features.shape[a] for a in reduction_axes]))
        if total_sum is None:
            total_sum = np.zeros_like(block_sum, dtype=np.float64)
            total_square_sum = np.zeros_like(block_square_sum, dtype=np.float64)
        total_sum += block_sum
        total_square_sum += block_square_sum
        total_values_per_radar += values_per_radar
    if total_sum is None or total_values_per_radar == 0:
        raise ValueError("No feature blocks were provided")
    mean = total_sum / total_values_per_radar
    variance = np.maximum(total_square_sum / total_values_per_radar - mean**2, 0.0)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32), total_values_per_radar


def per_radar_channel_moments(
    feature_blocks: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute train-only moments for ``(T, radar, channel, bin)`` blocks."""
    total_sum = None
    total_square_sum = None
    total_values_per_channel = 0
    for features in feature_blocks:
        features = np.asarray(features)
        if features.ndim != 4:
            raise ValueError(f"Expected (T,R,C,B), received {features.shape}")
        block_sum = features.sum(axis=(0, 3), dtype=np.float64)
        block_square_sum = np.square(features, dtype=np.float64).sum(
            axis=(0, 3), dtype=np.float64
        )
        values_per_channel = features.shape[0] * features.shape[3]
        if total_sum is None:
            total_sum = np.zeros_like(block_sum, dtype=np.float64)
            total_square_sum = np.zeros_like(block_square_sum, dtype=np.float64)
        total_sum += block_sum
        total_square_sum += block_square_sum
        total_values_per_channel += values_per_channel
    if total_sum is None or total_values_per_channel == 0:
        raise ValueError("No feature blocks were provided")
    mean = total_sum / total_values_per_channel
    variance = np.maximum(
        total_square_sum / total_values_per_channel - mean**2, 0.0
    )
    return (
        mean.astype(np.float32),
        np.sqrt(variance).astype(np.float32),
        total_values_per_channel,
    )


def normalize_per_radar(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Apply per-radar standardisation while preserving spatial structure."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if features.shape[1] != len(mean) or mean.shape != std.shape:
        raise ValueError("Normalisation statistics do not match the radar dimension")
    broadcast_shape = (1, len(mean)) + (1,) * (features.ndim - 2)
    normalized = (features - mean.reshape(broadcast_shape)) / (
        std.reshape(broadcast_shape) + epsilon
    )
    if not np.isfinite(normalized).all():
        raise ValueError("Normalisation produced non-finite values")
    return normalized.astype(np.float32, copy=False)


def normalize_per_radar_channel(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Standardise ``(T, radar, channel, bin)`` signed-I/Q features.

    The supplied ``(radar, channel)`` statistics must be estimated from the
    training windows only. Keeping this operation here ensures that
    preprocessing notebooks, model training, and final inference share the
    exact same broadcasting and numerical checks.
    """
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if features.ndim != 4:
        raise ValueError(f"Expected features shaped (T,R,C,B), received {features.shape}")
    if mean.shape != std.shape or mean.shape != features.shape[1:3]:
        raise ValueError("Normalisation statistics do not match radar/channel dimensions")
    if np.any(std <= 0):
        raise ValueError("All normalisation standard deviations must be positive")
    normalized = (features - mean[None, :, :, None]) / (
        std[None, :, :, None] + epsilon
    )
    if not np.isfinite(normalized).all():
        raise ValueError("Signed I/Q normalisation produced non-finite values")
    return normalized.astype(np.float32, copy=False)
