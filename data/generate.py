"""
Synthetic fNIRS Data Generator
-------------------------------
Generates realistic fNIRS (functional Near-Infrared Spectroscopy) signals
for two chromophores: HbO (oxygenated) and HbR (deoxygenated) hemoglobin.

Signal properties are based on published fNIRS literature:
- Scholkmann et al. (2014): "A review on continuous wave functional near-infrared
  spectroscopy and imaging instrumentation and methodology"
- Naseer & Hong (2015): "fNIRS-based brain-computer interfaces: a review"

Classes:
  0 = REST     (baseline, no task)
  1 = MENTAL   (mental arithmetic / cognitive load)
  2 = MOTOR    (finger tapping / motor imagery)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FNIRSConfig:
    fs: float = 10.0          # Sampling frequency (Hz) — typical fNIRS hardware
    n_channels: int = 16       # Number of source-detector pairs
    epoch_len: float = 10.0    # Epoch length (seconds)
    n_epochs_per_class: int = 60
    seed: int = 42


def _hemodynamic_response(t: np.ndarray, amplitude: float, delay: float = 5.0) -> np.ndarray:
    """
    Canonical hemodynamic response function (HRF).
    Models the BOLD-like response observed in fNIRS signals.
    Peak around 5–8 seconds post-stimulus.
    """
    from scipy.stats import gamma
    hrf = gamma.pdf(t, a=6, scale=1.0) - 0.35 * gamma.pdf(t, a=12, scale=1.0)
    hrf = hrf / (hrf.max() + 1e-8)
    return amplitude * hrf


def _mayer_wave(t: np.ndarray, amplitude: float = 0.02) -> np.ndarray:
    """Mayer waves: ~0.1 Hz oscillations from blood pressure regulation."""
    return amplitude * np.sin(2 * np.pi * 0.1 * t + np.random.uniform(0, 2 * np.pi))


def _respiratory(t: np.ndarray, amplitude: float = 0.015) -> np.ndarray:
    """Respiratory artifact: ~0.3 Hz."""
    return amplitude * np.sin(2 * np.pi * 0.3 * t + np.random.uniform(0, 2 * np.pi))


def generate_epoch(
    label: int,
    cfg: FNIRSConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one epoch of fNIRS data.

    Returns:
        hbo: (n_channels, n_samples) — oxygenated hemoglobin (μmol/L)
        hbr: (n_channels, n_samples) — deoxygenated hemoglobin (μmol/L)
    """
    n_samples = int(cfg.epoch_len * cfg.fs)
    t = np.linspace(0, cfg.epoch_len, n_samples)

    # Class-specific hemodynamic amplitudes
    # REST: near-zero task-related response
    # MENTAL: moderate HbO increase, moderate HbR decrease (prefrontal)
    # MOTOR: strong HbO increase in motor channels, strong HbR decrease
    class_params = {
        0: {"hbo_amp": 0.0,  "hbr_amp":  0.0,  "noise": 0.03},   # REST
        1: {"hbo_amp": 0.25, "hbr_amp": -0.10, "noise": 0.04},   # MENTAL
        2: {"hbo_amp": 0.45, "hbr_amp": -0.20, "noise": 0.05},   # MOTOR
    }
    p = class_params[label]

    hrf = _hemodynamic_response(t, amplitude=1.0)

    hbo = np.zeros((cfg.n_channels, n_samples))
    hbr = np.zeros((cfg.n_channels, n_samples))

    # Channels with task-related activation (spatially realistic)
    active_channels = rng.choice(cfg.n_channels, size=cfg.n_channels // 2, replace=False)

    for ch in range(cfg.n_channels):
        ch_scale = rng.uniform(0.7, 1.3)  # inter-channel variability
        task_factor = 1.0 if ch in active_channels else rng.uniform(0.0, 0.15)

        hbo[ch] = (
            p["hbo_amp"] * ch_scale * task_factor * hrf
            + _mayer_wave(t, amplitude=rng.uniform(0.01, 0.03))
            + _respiratory(t, amplitude=rng.uniform(0.01, 0.02))
            + rng.normal(0, p["noise"], n_samples)
        )
        hbr[ch] = (
            p["hbr_amp"] * ch_scale * task_factor * hrf
            + _mayer_wave(t, amplitude=rng.uniform(0.005, 0.015))
            + rng.normal(0, p["noise"] * 0.6, n_samples)
        )

    return hbo, hbr


def generate_dataset(cfg: FNIRSConfig = None) -> dict:
    """
    Generate a full labelled fNIRS dataset.

    Returns dict with keys:
        X_hbo:   (n_epochs, n_channels, n_samples)
        X_hbr:   (n_epochs, n_channels, n_samples)
        y:       (n_epochs,) — integer labels {0, 1, 2}
        labels:  class name mapping
        cfg:     config used
    """
    if cfg is None:
        cfg = FNIRSConfig()

    rng = np.random.default_rng(cfg.seed)
    classes = [0, 1, 2]  # REST, MENTAL, MOTOR

    X_hbo, X_hbr, y = [], [], []

    for label in classes:
        for _ in range(cfg.n_epochs_per_class):
            hbo, hbr = generate_epoch(label, cfg, rng)
            X_hbo.append(hbo)
            X_hbr.append(hbr)
            y.append(label)

    X_hbo = np.array(X_hbo)
    X_hbr = np.array(X_hbr)
    y = np.array(y)

    # Shuffle
    idx = rng.permutation(len(y))
    return {
        "X_hbo": X_hbo[idx],
        "X_hbr": X_hbr[idx],
        "y": y[idx],
        "labels": {0: "REST", 1: "MENTAL", 2: "MOTOR"},
        "cfg": cfg,
    }


if __name__ == "__main__":
    ds = generate_dataset()
    print(f"Dataset: {ds['X_hbo'].shape[0]} epochs, "
          f"{ds['X_hbo'].shape[1]} channels, "
          f"{ds['X_hbo'].shape[2]} samples/epoch")
    print(f"Class counts: { {ds['labels'][l]: int((ds['y']==l).sum()) for l in [0,1,2]} }")
