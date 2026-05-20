"""
fNIRS Signal Preprocessing
----------------------------
Standard preprocessing pipeline for fNIRS epochs:
  1. Bandpass filtering  — removes slow drift and high-freq noise
  2. Baseline correction — zero-mean per epoch
  3. Artifact rejection  — removes epochs with excessive amplitude

References:
  - Brigadoi et al. (2014): "Motion artifacts in functional near-infrared spectroscopy"
  - Scholkmann et al. (2014): Review of fNIRS methodology
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from dataclasses import dataclass


@dataclass
class PreprocConfig:
    fs: float = 10.0
    low_hz: float = 0.01     # High-pass cutoff: removes slow baseline drift
    high_hz: float = 0.5     # Low-pass cutoff: removes high-freq noise (cardiac ~1Hz filtered out)
    baseline_sec: float = 2.0  # Seconds at epoch start to use as baseline
    artifact_threshold: float = 3.0  # Z-score threshold for artifact rejection


def bandpass_filter(signal: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.
    sosfiltfilt applies filter forward and backward to eliminate phase shift.

    Args:
        signal: (n_channels, n_samples)
    Returns:
        filtered: same shape
    """
    nyq = cfg.fs / 2.0
    low = cfg.low_hz / nyq
    high = cfg.high_hz / nyq
    sos = butter(4, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, signal, axis=-1)


def baseline_correct(signal: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """
    Subtract the mean of the baseline window from each channel.
    Standard practice in event-related fNIRS analysis.

    Args:
        signal: (n_channels, n_samples)
    Returns:
        corrected: same shape
    """
    n_baseline = int(cfg.baseline_sec * cfg.fs)
    n_baseline = max(1, min(n_baseline, signal.shape[-1]))
    baseline_mean = signal[..., :n_baseline].mean(axis=-1, keepdims=True)
    return signal - baseline_mean


def is_artifact(epoch_hbo: np.ndarray, epoch_hbr: np.ndarray, threshold: float = 3.0) -> bool:
    """
    Simple amplitude-based artifact detection.
    Flags epoch if any channel exceeds threshold * global std.

    In real fNIRS: motion artifacts cause large transient spikes.
    More advanced: CBSI, PCA-based, or wavelet methods.
    """
    for sig in [epoch_hbo, epoch_hbr]:
        global_std = sig.std()
        if global_std == 0:
            continue
        z = np.abs(sig - sig.mean()) / global_std
        if z.max() > threshold:
            return True
    return False


def preprocess_dataset(dataset: dict, cfg: PreprocConfig = None) -> dict:
    """
    Apply full preprocessing pipeline to all epochs.

    Returns new dataset dict with:
        X_hbo, X_hbr: preprocessed signals
        y:             labels (artifact epochs removed)
        n_rejected:    count of rejected epochs
    """
    if cfg is None:
        cfg = PreprocConfig(fs=dataset["cfg"].fs)

    X_hbo_raw = dataset["X_hbo"]
    X_hbr_raw = dataset["X_hbr"]
    y_raw = dataset["y"]

    X_hbo_out, X_hbr_out, y_out = [], [], []
    n_rejected = 0

    for i in range(len(y_raw)):
        hbo = X_hbo_raw[i]  # (n_channels, n_samples)
        hbr = X_hbr_raw[i]

        # Step 1: Bandpass filter
        hbo = bandpass_filter(hbo, cfg)
        hbr = bandpass_filter(hbr, cfg)

        # Step 2: Baseline correction
        hbo = baseline_correct(hbo, cfg)
        hbr = baseline_correct(hbr, cfg)

        # Step 3: Artifact rejection
        if is_artifact(hbo, hbr, cfg.artifact_threshold):
            n_rejected += 1
            continue

        X_hbo_out.append(hbo)
        X_hbr_out.append(hbr)
        y_out.append(y_raw[i])

    print(f"Preprocessing complete: {len(y_out)} epochs kept, {n_rejected} rejected.")

    return {
        "X_hbo": np.array(X_hbo_out),
        "X_hbr": np.array(X_hbr_out),
        "y": np.array(y_out),
        "labels": dataset["labels"],
        "cfg": dataset["cfg"],
        "n_rejected": n_rejected,
    }


if __name__ == "__main__":
    from data.generate import generate_dataset
    raw = generate_dataset()
    processed = preprocess_dataset(raw)
    print(f"Shape after preprocessing: {processed['X_hbo'].shape}")
