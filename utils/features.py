"""
fNIRS Feature Extraction
--------------------------
Extracts interpretable features from preprocessed HbO/HbR epochs.

Feature families:
  1. Statistical (time-domain)  — mean, variance, skew, kurtosis, slope
  2. Spectral (frequency-domain) — power in Mayer wave / respiratory bands
  3. Coupling                   — HbO-HbR correlation (CBSI index)
  4. Spatial                    — channel asymmetry (left vs right hemispheres)

This mimics what a real BCI decoding pipeline does before classification.
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


def _channel_stats(signal: np.ndarray) -> np.ndarray:
    """
    Per-channel statistical features.
    signal: (n_channels, n_samples)
    Returns: (n_channels * 5,)
    """
    mean = signal.mean(axis=-1)
    var = signal.var(axis=-1)
    sk = skew(signal, axis=-1)
    kurt = kurtosis(signal, axis=-1)
    # Linear slope via least-squares — captures hemodynamic trend direction
    t = np.arange(signal.shape[-1])
    slope = np.array([
        np.polyfit(t, signal[ch], 1)[0] for ch in range(signal.shape[0])
    ])
    return np.concatenate([mean, var, sk, kurt, slope])


def _spectral_power(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Band power in physiologically meaningful fNIRS frequency bands.
    signal: (n_channels, n_samples)

    Bands:
      Very low freq (VLF): 0.01–0.04 Hz — slow vasomotion
      Mayer wave:          0.07–0.13 Hz — blood pressure oscillation
      Respiratory:         0.15–0.40 Hz — breathing

    Returns: (n_channels * 3,)
    """
    bands = {
        "vlf":        (0.01, 0.04),
        "mayer":      (0.07, 0.13),
        "respiratory":(0.15, 0.40),
    }
    band_powers = []
    for ch in range(signal.shape[0]):
        freqs, psd = welch(signal[ch], fs=fs, nperseg=min(signal.shape[-1], int(fs * 5)))
        ch_powers = []
        for (flo, fhi) in bands.values():
            mask = (freqs >= flo) & (freqs <= fhi)
            ch_powers.append(psd[mask].mean() if mask.any() else 0.0)
        band_powers.extend(ch_powers)
    return np.array(band_powers)


def _hbo_hbr_coupling(hbo: np.ndarray, hbr: np.ndarray) -> np.ndarray:
    """
    HbO–HbR coupling features per channel.

    Key insight: during genuine neural activation, HbO increases and HbR decreases
    (neurovascular coupling). Their correlation should be negative during tasks.
    The Correlation-Based Signal Improvement (CBSI) index exploits this.

    Returns: (n_channels * 2,)  — [correlation, cbsi_ratio]
    """
    correlations = []
    cbsi_ratios = []
    for ch in range(hbo.shape[0]):
        h = hbo[ch]
        r = hbr[ch]
        if h.std() > 1e-10 and r.std() > 1e-10:
            corr = np.corrcoef(h, r)[0, 1]
        else:
            corr = 0.0
        # CBSI ratio: std(HbO) / std(HbR) — ~1.0 at rest, differs during activation
        cbsi = h.std() / (r.std() + 1e-10)
        correlations.append(corr)
        cbsi_ratios.append(cbsi)
    return np.concatenate([np.array(correlations), np.array(cbsi_ratios)])


def _spatial_asymmetry(signal: np.ndarray) -> np.ndarray:
    """
    Approximate left/right hemispheric asymmetry.
    Assumes channels are ordered: even = left, odd = right (common montage convention).

    Returns: (n_channels // 2,)
    """
    left = signal[::2]    # even indices
    right = signal[1::2]  # odd indices
    n = min(left.shape[0], right.shape[0])
    return (left[:n].mean(axis=-1) - right[:n].mean(axis=-1))


def extract_features(hbo: np.ndarray, hbr: np.ndarray, fs: float = 10.0) -> np.ndarray:
    """
    Extract full feature vector for one epoch.

    Args:
        hbo: (n_channels, n_samples)
        hbr: (n_channels, n_samples)
        fs:  sampling frequency

    Returns:
        features: 1D numpy array
    """
    feats = []

    # 1. Statistical features for HbO and HbR
    feats.append(_channel_stats(hbo))
    feats.append(_channel_stats(hbr))

    # 2. Spectral power for HbO
    feats.append(_spectral_power(hbo, fs))

    # 3. HbO-HbR coupling
    feats.append(_hbo_hbr_coupling(hbo, hbr))

    # 4. Spatial asymmetry
    feats.append(_spatial_asymmetry(hbo))

    return np.concatenate(feats)


def build_feature_matrix(dataset: dict) -> tuple:
    """
    Build (X, y) feature matrix from preprocessed dataset.

    Returns:
        X: (n_epochs, n_features)
        y: (n_epochs,)
        feature_dim: int
    """
    fs = dataset["cfg"].fs
    X_hbo = dataset["X_hbo"]
    X_hbr = dataset["X_hbr"]
    y = dataset["y"]

    features = []
    for i in range(len(y)):
        f = extract_features(X_hbo[i], X_hbr[i], fs)
        features.append(f)

    X = np.array(features)
    print(f"Feature matrix: {X.shape[0]} epochs × {X.shape[1]} features")
    return X, y


if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.generate import generate_dataset
    from utils.preprocess import preprocess_dataset

    raw = generate_dataset()
    proc = preprocess_dataset(raw)
    X, y = build_feature_matrix(proc)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Any NaN: {np.isnan(X).any()}")
