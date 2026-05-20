# fNIRS Mental State Decoder
### A BCI signal processing pipeline for cognitive state classification

---

## What This Is

A full end-to-end pipeline that decodes **mental state** (REST / MENTAL / MOTOR) from
**fNIRS (functional Near-Infrared Spectroscopy)** signals — the modality used
in cerebral blood flow wearables and neuroimaging research.

The pipeline mirrors what a real neuroimaging ML system does:

```
Raw fNIRS signal
    → Bandpass filter (0.01–0.5 Hz)
    → Baseline correction
    → Artifact rejection (amplitude z-score)
    → Feature extraction (statistical + spectral + HbO/HbR coupling)
    → Classification (LDA / RF / SVM / Gradient Boosting)
    → Decoded mental state
```

---

## Why fNIRS?

fNIRS measures **hemodynamic response** — changes in oxygenated (HbO) and
deoxygenated (HbR) hemoglobin — using near-infrared light through the skull.
It's non-invasive, wearable, and directly measures cerebral blood flow.

**Signal characteristics modeled here:**
- **Hemodynamic response function (HRF):** ~5–8s peak post-stimulus
- **Mayer waves:** ~0.1 Hz blood pressure oscillations (physiological noise)
- **Respiratory artifacts:** ~0.3 Hz breathing-related signal
- **HbO/HbR coupling (CBSI):** during activation, HbO ↑ and HbR ↓ (neurovascular coupling)

---

## Project Structure

```
fnirs_pipeline/
├── data/
│   └── generate.py              # Synthetic fNIRS data with realistic physiology
├── utils/
│   ├── preprocess.py            # Bandpass filter, baseline correction, artifact rejection
│   ├── features.py              # Statistical, spectral, coupling, spatial features
│   └── artifact_removal.py      # ICA + CBSI software artifact removal
├── models/
│   └── classify.py              # LDA, RF, SVM, GradBoost — cross-validated
├── notebooks/
│   └── 01_artifact_removal_deep_dive.ipynb
├── app.py                       # Streamlit interactive demo
├── requirements.txt
├── README.md
├── LEARNING_GUIDE.md            # Beginner-friendly step-by-step walkthrough
└── APP_WALKTHROUGH.md           # Visual guide to the Streamlit app
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Run the interactive demo
streamlit run app.py

# Or run the pipeline from the command line
python models/classify.py
```

---

## Results

| Model | Accuracy (5-fold CV) | F1-Macro |
|---|---|---|
| LDA | 0.938 ± 0.040 | 0.830 |
| Random Forest | 0.938 ± 0.001 | 0.643 |
| SVM (RBF) | 0.901 ± 0.030 | 0.616 |
| Gradient Boosting | 0.888 ± 0.048 | 0.611 |

**LDA performs best** — consistent with BCI literature showing linear classifiers
often outperform complex models on small-N neuroimaging datasets
(Blankertz et al., 2011; Lotte et al., 2018).

---

## Key Concepts Implemented

**Preprocessing:**
- Zero-phase Butterworth bandpass (prevents phase distortion in hemodynamic signals)
- Baseline correction (standard for event-related fNIRS)
- Amplitude-based artifact rejection (motion artifact proxy)

**Features:**
- Per-channel statistics: mean, variance, skew, kurtosis, linear slope
- Band power: VLF (0.01–0.04 Hz), Mayer wave (0.07–0.13 Hz), respiratory (0.15–0.40 Hz)
- **CBSI (Correlation-Based Signal Improvement):** exploits the physiological anti-correlation
  between HbO and HbR during neural activation (Cui et al., 2010)
- Hemispheric asymmetry: left/right channel differences

**Artifact removal** (`utils/artifact_removal.py`):
- **CBSI (Correlation-Based Signal Improvement):** removes systemic noise by enforcing HbO/HbR anti-correlation (Cui et al., 2010)
- **ICA + Wavelet decomposition:** creates virtual channels from a single sensor, identifies and removes artifact components by kurtosis
- **Combined pipeline:** ICA first (broadband artifacts), then CBSI (systemic hemodynamics)

**Next steps for real data:**
- Channel selection / spatial filtering
- Subject-independent (cross-subject) validation
- Deep learning: CNN or Transformer on raw signals

---

## References

- Scholkmann et al. (2014). "A review on continuous wave functional near-infrared spectroscopy"
- Naseer & Hong (2015). "fNIRS-based brain-computer interfaces: a review"
- Cui et al. (2010). "Functional near infrared spectroscopy (NIRS) signal improvement"
- Lotte et al. (2018). "A review of classification algorithms for EEG-based BCI"
- Blankertz et al. (2011). "Single-trial analysis and classification of ERP components"

---

*Nancy Tyagi*
