"""
fNIRS Artifact Removal: ICA and Anti-Correlation Method
---------------------------------------------------------
Implements two software-based approaches to removing scalp/systemic noise
from fNIRS signals when no physical short-separation channel is available.

This is the key engineering challenge for single-sensor wearables: one sensor
on the scalp with no reference channel to subtract scalp hemodynamics.

METHODS IMPLEMENTED
-------------------
1. Single-Channel ICA (FastICA with wavelet decomposition)
   - Decomposes signal into statistically independent components
   - Identifies and removes components with non-neural statistics
   - Based on: Hyvarinen & Oja (2000), FastICA algorithm

2. Anti-Correlation (CBSI) Method
   - Exploits physiological anti-correlation between HbO and HbR
   - True brain activation: HbO ↑, HbR ↓  (anti-correlated)
   - Systemic noise:        HbO ↑, HbR ↑  (co-correlated)
   - Based on: Cui et al. (2010), Scholkmann et al. (2014)

REFERENCES
----------
- Hyvarinen & Oja (2000): "Independent component analysis: algorithms and applications"
- Cui et al. (2010): "Functional near infrared spectroscopy (NIRS) signal improvement
  based on negative correlation between oxygenated and deoxygenated hemoglobin dynamics"
- Scholkmann et al. (2014): "A review on continuous wave fNIRS methodology"
- Brigadoi et al. (2014): "Motion artifacts in functional near-infrared spectroscopy:
  a comparison of motion correction techniques"
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt  # noqa: F401
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
import pywt
from dataclasses import dataclass
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Single-Channel ICA via Wavelet Decomposition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ICAConfig:
    fs: float = 10.0
    n_components: int = 4       # Number of "virtual channels" from wavelet decomp
                                 # (DWT max level for 100-sample epochs at db4 = 3 → 4 arrays)
    n_ica_components: int = 4   # ICA components to extract
    kurtosis_threshold: float = 1.5  # Components with |kurt| > threshold = artifact
    max_iter: int = 500
    random_state: int = 42


def _wavelet_decompose(signal_1d: np.ndarray, n_components: int, fs: float = 10.0) -> np.ndarray:
    """
    Decompose a single channel into multiple pseudo-channels using
    Discrete Wavelet Transform (DWT) detail and approximation coefficients.

    WHY: ICA needs multiple input channels to separate sources. Since a single-sensor
    wearable has one channel, we use wavelets to create "virtual channels" — each one
    capturing the signal's behavior at a different frequency band.

    DWT levels map to frequency bands (at fs=10 Hz, db4 wavelet):
      Level 1 detail: 2.5–5.0 Hz  (cardiac, motion)
      Level 2 detail: 1.25–2.5 Hz
      Level 3 detail: 0.63–1.25 Hz (respiratory edge)
      Level 4 detail: 0.31–0.63 Hz (respiratory)
      Level 5 detail: 0.16–0.31 Hz (Mayer wave edge)
      Level 6 approx: 0.0–0.16 Hz  (VLF, slow hemodynamics)

    Args:
        signal_1d: (n_samples,)
        n_components: number of decomposition levels to use

    Returns:
        pseudo_channels: (n_components, n_samples) — each row is one frequency band
    """
    n_samples = len(signal_1d)
    wavelet = 'db4'  # Daubechies 4 — good for smooth hemodynamic signals
    max_level = pywt.dwt_max_level(n_samples, wavelet)
    n_levels = min(n_components, max_level)

    # Full wavedec decomposition
    coeffs = pywt.wavedec(signal_1d, wavelet, level=n_levels)
    # coeffs[0] = approximation (lowest freq), coeffs[1..] = details (high to low freq)

    pseudo = np.zeros((len(coeffs), n_samples))
    for i, c in enumerate(coeffs):
        # Reconstruct each sub-band back to original length for comparability
        # Zero out all other levels and reconstruct
        zero_coeffs = [np.zeros_like(cc) for cc in coeffs]
        zero_coeffs[i] = c
        reconstructed = pywt.waverec(zero_coeffs, wavelet)
        # Trim or pad to n_samples (waverec can produce n+1 samples)
        reconstructed = reconstructed[:n_samples]
        if len(reconstructed) < n_samples:
            reconstructed = np.pad(reconstructed, (0, n_samples - len(reconstructed)))
        pseudo[i] = reconstructed

    return pseudo[:n_components]


def _identify_artifact_components(
    components: np.ndarray,
    kurtosis_threshold: float
) -> np.ndarray:
    """
    Identify which ICA components are likely artifacts vs neural signal.

    LOGIC:
    - Neural (hemodynamic) signals are slow, smooth, and Gaussian-like → low kurtosis
    - Motion artifacts and cardiac pulses are spiky, impulsive → high kurtosis
    - Mayer waves are very regular sinusoids → negative kurtosis

    A simple but effective heuristic used in real fNIRS pipelines.
    More sophisticated: train a classifier on labeled component features.

    Args:
        components: (n_components, n_samples)
        kurtosis_threshold: absolute kurtosis above this = artifact

    Returns:
        artifact_mask: boolean array (n_components,), True = artifact
    """
    kurt_vals = kurtosis(components, axis=1)
    artifact_mask = np.abs(kurt_vals) > kurtosis_threshold
    return artifact_mask, kurt_vals


def apply_ica_single_channel(
    signal_1d: np.ndarray,
    cfg: ICAConfig = None,
    return_diagnostics: bool = False
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Single-channel ICA artifact removal pipeline.

    Pipeline:
      1. Wavelet decomposition → pseudo-multichannel matrix
      2. FastICA → independent components
      3. Kurtosis-based artifact identification
      4. Zero out artifact components
      5. Reconstruct clean signal via ICA mixing matrix inverse

    Args:
        signal_1d: (n_samples,) — one channel of raw fNIRS
        cfg: ICAConfig
        return_diagnostics: if True, return intermediate results

    Returns:
        clean_signal: (n_samples,)
        diagnostics: dict with components, kurtosis values, artifact mask (if requested)
    """
    if cfg is None:
        cfg = ICAConfig()

    # Step 1: Create pseudo-multichannel representation via wavelets
    pseudo_channels = _wavelet_decompose(signal_1d, cfg.n_components, cfg.fs)
    # Shape: (n_components, n_samples) → transpose for sklearn: (n_samples, n_components)
    X = pseudo_channels.T

    # Step 2: FastICA decomposition
    # FastICA maximizes non-Gaussianity to find independent sources
    # The "cocktail party" step — unmixing the mixed signals
    ica = FastICA(
        n_components=min(cfg.n_ica_components, cfg.n_components),
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        whiten='unit-variance',
    )
    try:
        components_T = ica.fit_transform(X)   # (n_samples, n_ica_components)
    except Exception:
        # ICA can fail to converge on very clean/flat signals — return as-is
        if return_diagnostics:
            return signal_1d.copy(), {}
        return signal_1d.copy()

    components = components_T.T  # (n_ica_components, n_samples)

    # Step 3: Identify artifact components by kurtosis
    artifact_mask, kurt_vals = _identify_artifact_components(
        components, cfg.kurtosis_threshold
    )

    # Safety: if ALL components flagged, keep the least-artifactual one
    # (returning zeros would blow up the scale factor)
    if artifact_mask.all():
        best = np.argmin(np.abs(kurt_vals))
        artifact_mask[best] = False

    # Step 4: Zero out artifact components
    components_clean = components.copy()
    components_clean[artifact_mask] = 0.0

    # Step 5: Reconstruct using ICA mixing matrix
    # ICA model: X = S @ A.T + mean  →  X_clean = S_clean @ A.T + mean
    mixing = ica.mixing_          # (n_components, n_ica_components)
    mean = ica.mean_              # (n_components,)
    reconstructed = (components_clean.T @ mixing.T) + mean  # (n_samples, n_components)

    # Project back to 1D: take the first pseudo-channel's reconstruction
    # (weighted average would also work)
    clean_signal = reconstructed[:, 0]

    # Preserve scale — ICA changes amplitude
    scale = np.std(signal_1d) / (np.std(clean_signal) + 1e-10)
    clean_signal = clean_signal * scale

    if return_diagnostics:
        diag = {
            "components": components,
            "kurt_vals": kurt_vals,
            "artifact_mask": artifact_mask,
            "n_removed": int(artifact_mask.sum()),
            "pseudo_channels": pseudo_channels,
        }
        return clean_signal, diag

    return clean_signal, None


def apply_ica_multichannel(
    signal: np.ndarray,
    cfg: ICAConfig = None,
    return_diagnostics: bool = False
) -> Tuple[np.ndarray, list]:
    """
    Apply single-channel ICA to each channel independently.

    Args:
        signal: (n_channels, n_samples)
    Returns:
        clean: (n_channels, n_samples)
        diagnostics: list of per-channel diagnostic dicts
    """
    if cfg is None:
        cfg = ICAConfig()

    clean = np.zeros_like(signal)
    diagnostics = []

    for ch in range(signal.shape[0]):
        clean_ch, diag = apply_ica_single_channel(
            signal[ch], cfg, return_diagnostics=return_diagnostics
        )
        clean[ch] = clean_ch
        diagnostics.append(diag)

    return clean, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Anti-Correlation (CBSI) Method
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CBSIConfig:
    """
    Configuration for Correlation-Based Signal Improvement (CBSI).

    CBSI exploits the fundamental physiology of neurovascular coupling:
    - True neural activation:  HbO ↑  HbR ↓  (anti-correlated)
    - Systemic/scalp noise:    HbO ↑  HbR ↑  (co-correlated — both go up together)

    The algorithm enforces this anti-correlation as a constraint to
    separate brain signal from scalp hemodynamics.

    Reference: Cui et al. (2010) NeuroImage
    """
    alpha: float = None   # Scaling factor (None = estimate from data as std ratio)
    window_sec: float = None  # If set, use sliding window CBSI instead of global


def apply_cbsi(
    hbo: np.ndarray,
    hbr: np.ndarray,
    cfg: CBSIConfig = None,
    return_diagnostics: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """
    Correlation-Based Signal Improvement (CBSI) — Cui et al. (2010).

    MATH:
      Let α = std(HbO) / std(HbR)  [channel-wise scaling factor]

      Corrected HbO:  HbO_c = 0.5 * (HbO  - α * HbR)
      Corrected HbR:  HbR_c = 0.5 * (HbR  - (1/α) * HbO)

    INTUITION:
      If both HbO and HbR increase together (systemic noise), the subtraction
      cancels the shared component. If HbO rises while HbR falls (brain signal),
      the subtraction reinforces the difference — signal is preserved or enhanced.

    Args:
        hbo: (n_channels, n_samples)
        hbr: (n_channels, n_samples)
        cfg: CBSIConfig

    Returns:
        hbo_clean: (n_channels, n_samples)
        hbr_clean: (n_channels, n_samples)
        diagnostics: dict with per-channel alpha, correlation before/after
    """
    if cfg is None:
        cfg = CBSIConfig()

    n_channels = hbo.shape[0]
    hbo_clean = np.zeros_like(hbo)
    hbr_clean = np.zeros_like(hbr)
    alphas = np.zeros(n_channels)
    corr_before = np.zeros(n_channels)
    corr_after = np.zeros(n_channels)

    for ch in range(n_channels):
        h = hbo[ch]
        r = hbr[ch]

        std_h = np.std(h)
        std_r = np.std(r)

        # Compute alpha: ratio of standard deviations
        # This normalizes for the fact that HbO typically has larger amplitude than HbR
        # Cap alpha to [0.1, 10] to prevent blowup when either channel is near-flat
        if cfg.alpha is not None:
            alpha = cfg.alpha
        else:
            alpha = std_h / (std_r + 1e-10)
            alpha = float(np.clip(alpha, 0.1, 10.0))

        alphas[ch] = alpha

        # Correlation before CBSI (diagnostic)
        if std_h > 1e-10 and std_r > 1e-10:
            corr_before[ch] = np.corrcoef(h, r)[0, 1]
        else:
            corr_before[ch] = 0.0

        # CBSI correction
        hbo_clean[ch] = 0.5 * (h - alpha * r)
        hbr_clean[ch] = 0.5 * (r - (1.0 / (alpha + 1e-10)) * h)

        # Correlation after CBSI (should be more negative = more anti-correlated)
        std_hc = np.std(hbo_clean[ch])
        std_rc = np.std(hbr_clean[ch])
        if std_hc > 1e-10 and std_rc > 1e-10:
            corr_after[ch] = np.corrcoef(hbo_clean[ch], hbr_clean[ch])[0, 1]
        else:
            corr_after[ch] = 0.0

    diag = {
        "alphas": alphas,
        "corr_before": corr_before,
        "corr_after": corr_after,
        "mean_corr_before": float(corr_before.mean()),
        "mean_corr_after": float(corr_after.mean()),
    } if return_diagnostics else None

    return hbo_clean, hbr_clean, diag


def apply_cbsi_dataset(dataset: dict, cfg: CBSIConfig = None) -> dict:
    """
    Apply CBSI to all epochs in a preprocessed dataset.

    Returns new dataset dict with cleaned HbO and HbR.
    """
    if cfg is None:
        cfg = CBSIConfig()

    X_hbo = dataset["X_hbo"]
    X_hbr = dataset["X_hbr"]
    y = dataset["y"]

    hbo_out = np.zeros_like(X_hbo)
    hbr_out = np.zeros_like(X_hbr)

    for i in range(len(y)):
        hbo_c, hbr_c, _ = apply_cbsi(X_hbo[i], X_hbr[i], cfg)
        hbo_out[i] = hbo_c
        hbr_out[i] = hbr_c

    return {
        **dataset,
        "X_hbo": hbo_out,
        "X_hbr": hbr_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Combined pipeline
# ─────────────────────────────────────────────────────────────────────────────

def full_artifact_removal_pipeline(
    hbo: np.ndarray,
    hbr: np.ndarray,
    fs: float = 10.0,
    use_ica: bool = True,
    use_cbsi: bool = True,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Combined ICA + CBSI pipeline for one epoch.

    ORDER MATTERS:
      ICA first — removes broadband artifacts (motion, cardiac)
      CBSI second — removes systemic hemodynamics using HbO/HbR relationship
      This ordering is recommended by Brigadoi et al. (2014).

    Args:
        hbo: (n_channels, n_samples)
        hbr: (n_channels, n_samples)
        fs: sampling frequency
        use_ica: apply ICA
        use_cbsi: apply CBSI

    Returns:
        hbo_clean, hbr_clean, diagnostics
    """
    diag = {}

    # Step 1: ICA
    if use_ica:
        ica_cfg = ICAConfig(fs=fs)
        hbo_ica, ica_diags_hbo = apply_ica_multichannel(hbo, ica_cfg, return_diagnostics)
        hbr_ica, ica_diags_hbr = apply_ica_multichannel(hbr, ica_cfg, return_diagnostics)
        # Safety: clip to 5 std to prevent ICA scale explosion before CBSI
        for arr in [hbo_ica, hbr_ica]:
            std = arr.std()
            if std > 0:
                np.clip(arr, -5 * std, 5 * std, out=arr)
        if return_diagnostics:
            diag["ica_hbo"] = ica_diags_hbo
            diag["ica_hbr"] = ica_diags_hbr
    else:
        hbo_ica, hbr_ica = hbo.copy(), hbr.copy()

    # Step 2: CBSI
    if use_cbsi:
        cbsi_cfg = CBSIConfig()
        hbo_clean, hbr_clean, cbsi_diag = apply_cbsi(
            hbo_ica, hbr_ica, cbsi_cfg, return_diagnostics
        )
        if return_diagnostics:
            diag["cbsi"] = cbsi_diag
    else:
        hbo_clean, hbr_clean = hbo_ica, hbr_ica

    return hbo_clean, hbr_clean, diag


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.generate import generate_dataset
    from utils.preprocess import preprocess_dataset

    raw = generate_dataset()
    proc = preprocess_dataset(raw)

    # Test on one epoch
    hbo = proc["X_hbo"][0]
    hbr = proc["X_hbr"][0]

    print(f"Input  HbO: mean={hbo.mean():.4f}  std={hbo.std():.4f}")

    # CBSI
    hbo_c, hbr_c, diag = apply_cbsi(hbo, hbr, return_diagnostics=True)
    print(f"CBSI   HbO: mean={hbo_c.mean():.4f}  std={hbo_c.std():.4f}")
    print(f"  Corr before: {diag['mean_corr_before']:.3f}")
    print(f"  Corr after:  {diag['mean_corr_after']:.3f}")

    # ICA
    hbo_i, ica_diags = apply_ica_multichannel(hbo, return_diagnostics=True)
    print(f"\nICA    HbO: mean={hbo_i.mean():.4f}  std={hbo_i.std():.4f}")
    n_removed = sum(d["n_removed"] for d in ica_diags if d)
    print(f"  Components removed: {n_removed} across {hbo.shape[0]} channels")
