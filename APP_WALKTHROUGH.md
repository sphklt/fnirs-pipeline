# Streamlit App — Visual Walkthrough

A screen-by-screen guide to the fNIRS Mental State Decoder app.
Run it with: `streamlit run app.py`

---

## Screen 1 — App on First Load

When you open the app, you see the title, sidebar controls, and an empty main area
waiting for you to run the pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🧠 fNIRS Mental State Decoder                                               │
│ A prototype BCI pipeline decoding cognitive state from fNIRS signals.       │
├───────────────────┬─────────────────────────────────────────────────────────┤
│ SIDEBAR           │                                                         │
│                   │                                                         │
│ ⚙️ Pipeline Config │         (empty — run the pipeline to see results)      │
│                   │                                                         │
│ Epochs per class  │                                                         │
│ ●───────── 60     │                                                         │
│                   │                                                         │
│ Channels          │                                                         │
│ [4][8][16✓][32]   │                                                         │
│                   │                                                         │
│ Sampling rate(Hz) │                                                         │
│ [5] [10✓] [20]    │                                                         │
│                   │                                                         │
│ Epoch length (s)  │                                                         │
│ ●───────── 10     │                                                         │
│                   │                                                         │
│ Artifact thresh(σ)│                                                         │
│ ●───────── 3.0    │                                                         │
│                   │                                                         │
│ [▶ Run Pipeline]  │                                                         │
│ ─────────────     │                                                         │
│ Live Decode       │                                                         │
│ [REST ▾]          │                                                         │
│ [🔍 Decode Epoch] │                                                         │
└───────────────────┴─────────────────────────────────────────────────────────┘
```

**What each sidebar control does:**

| Control | Default | Effect |
|---|---|---|
| Epochs per class | 60 | How many trials per class (60×3 = 180 total) |
| Channels | 16 | Number of fNIRS sensors simulated |
| Sampling rate | 10 Hz | Measurements per second |
| Epoch length | 10s | Duration of each trial |
| Artifact threshold | 3.0σ | How aggressive artifact rejection is — lower = more epochs rejected |

---

## Screen 2 — While the Pipeline Runs

After clicking **▶ Run Pipeline**, four sequential spinners appear as each stage completes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ⟳  Generating synthetic fNIRS data...                                    │
│                                                                             │
│   ⟳  Preprocessing (bandpass filter + baseline correction + artifact        │
│       rejection)...                                                         │
│                                                                             │
│   ⟳  Extracting features...                                                │
│                                                                             │
│   ⟳  Training & evaluating classifiers...                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Each spinner disappears when that stage finishes. The whole run takes a few seconds.

---

## Screen 3 — Results: Metrics Row

Once the pipeline completes, four summary metrics appear at the top of the main area.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  ┌──────────┐ │
│  │ Epochs       │  │ After artifact   │  │ Feature        │  │ Best     │ │
│  │ generated    │  │ rejection        │  │ dimensions     │  │ model    │ │
│  │              │  │                  │  │                │  │ accuracy │ │
│  │     180      │  │      174         │  │      248       │  │  93.5%   │ │
│  │              │  │   ↓ -6 removed   │  │                │  │          │ │
│  └──────────────┘  └──────────────────┘  └────────────────┘  └──────────┘ │
│                                                                             │
│ ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  📡 Raw Signal  🔬 Preprocessed  🧹 Artifact Removal  📊 Model  🗺️ Matrix  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Reading the metrics:**
- **Epochs generated** — total trials created (epochs_per_class × 3)
- **After artifact rejection** — how many survived cleaning (delta shows how many were thrown away)
- **Feature dimensions** — size of the feature vector per epoch (~248)
- **Best model accuracy** — held-out test set accuracy of the best classifier

---

## Screen 4 — Tab 1: Raw Signal

Shows the raw, uncleaned fNIRS signal for one example epoch from each class.
Three stacked panels — one per mental state.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Raw fNIRS Signal — Example Epochs                                           │
│                                                                             │
│  REST   │ HbO ──── (red)   HbR ──── (blue)                                 │
│         │                                                                   │
│  0.06 ┤ ~∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (HbO — flat, noisy)     │
│  0.00 ┤ ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈  (HbR — flat, noisy)     │
│ -0.06 ┤                                                                     │
│                                                                             │
│  MENTAL │                                                                   │
│         │                        ╭────╮                                     │
│  0.30 ┤                       ╭─╯    ╰──╮  (HbO — rises, peaks ~5s)       │
│  0.00 ┤ ──────────────────────╯         ╰───                               │
│ -0.10 ┤ ───────────────────╮                ╭──  (HbR — dips oppositely)  │
│        │                   ╰────────────────╯                               │
│                                                                             │
│  MOTOR  │                                                                   │
│         │                     ╭──────╮                                      │
│  0.50 ┤                    ╭─╯      ╰──╮  (HbO — stronger rise)           │
│  0.00 ┤ ───────────────────╯           ╰────                               │
│ -0.20 ┤ ──────────────╮                    ╭──  (HbR — deeper dip)        │
│        │               ╰────────────────────╯                               │
│         └───────────────────────────────────────▶ Time (s)                 │
│              0        5        10                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What to notice:**
- REST has no clear pattern — just random noise with slow Mayer wave wobbles
- MENTAL shows a clear HbO rise and HbR dip starting around second 2–3
- MOTOR shows a stronger version of the same pattern (larger amplitude)
- All signals have visible noise riding on top of the brain signal — this is what preprocessing removes

---

## Screen 5 — Tab 2: Preprocessed

Same structure as Tab 1, but after bandpass filtering and baseline correction.
The dashed horizontal line shows the zero baseline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ After Preprocessing (bandpass + baseline correction)                        │
│                                                                             │
│  REST   │                                                                   │
│  0.04 ┤ ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (smaller noise, centred)     │
│  0.00 ┤- - - - - - - - - - - - - - - - - - -  (baseline)                  │
│ -0.04 ┤ ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈                              │
│                                                                             │
│  MENTAL │                                                                   │
│         │                     ╭────╮                                        │
│  0.25 ┤                    ╭─╯    ╰──╮   (HbO — cleaner rise)            │
│  0.00 ┤- ──────────────────╯- - - - -╰──  (baseline at zero)              │
│ -0.10 ┤ ─────────────────╮              ╭  (HbR — cleaner dip)            │
│        │                  ╰──────────────╯                                  │
│                                                                             │
│  MOTOR  │                                                                   │
│         │                  ╭──────────╮                                     │
│  0.45 ┤                 ╭─╯          ╰──╮  (stronger, cleaner)            │
│  0.00 ┤- ───────────────╯ - - - - - - - ╰───                              │
│ -0.20 ┤ ────────────╮                      ╭  (deeper, cleaner dip)       │
│        │              ╰────────────────────╯                                │
│         └───────────────────────────────────────▶ Time (s)                 │
│              0        5        10                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What changed vs Tab 1:**
- All channels now start at zero (baseline correction)
- High-frequency jitter is reduced (bandpass filter)
- The HRF shape in MENTAL and MOTOR is cleaner and easier to see
- REST is now clearly near-flat — the drift is gone

---

## Screen 6 — Tab 3: Artifact Removal

The most interactive tab. You choose a cleaning method and a mental state to visualise.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Software Artifact Removal — ICA + Anti-Correlation (CBSI)                  │
│                                                                             │
│ ℹ️  Single-sensor wearable has no reference channel. These algorithms       │
│    remove scalp noise using signal structure alone.                         │
│                                                                             │
│ Show cleaning method:                                                       │
│ ● CBSI (Anti-Correlation)   ○ ICA (Wavelet)   ○ ICA + CBSI (Full pipeline)│
│                                                                             │
│ Mental state to visualize: [MOTOR ▾]                                       │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐             │
│  │ HbO-HbR corr     │  │ HbO-HbR corr     │  │ Method       │             │
│  │ before           │  │ after            │  │              │             │
│  │   -0.142         │  │   -1.000         │  │    CBSI      │             │
│  │                  │  │   ↓ -0.858       │  │              │             │
│  └──────────────────┘  └──────────────────┘  └──────────────┘             │
│                                                                             │
│  Panel 1 — Raw signal (channel average)                                    │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 0.3 ┤     ╭──────╮            HbO raw ────  (red)              │       │
│  │ 0.0 ┤─────╯      ╰─────       HbR raw ────  (blue)             │       │
│  │-0.1 ┤ ╮            ╭───                                         │       │
│  │     │  ╰──────────╯                                             │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Panel 2 — After CBSI (cleaned)                                            │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 0.2 ┤    ╭───────╮           HbO cleaned ──── (orange)         │       │
│  │ 0.0 ┤────╯       ╰────       HbR cleaned ──── (green)          │       │
│  │-0.1 ┤ ╮            ╭──                                          │       │
│  │     │  ╰───────────╯                                            │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Panel 3 — Noise removed (HbO raw − cleaned)                               │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 0.1 ┤ ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (purple — the noise that was    │       │
│  │ 0.0 ┤- - - - - - - - - - - - -   subtracted away)              │       │
│  │-0.1 ┤                                                           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│ CBSI insight: After correction, HbO-HbR correlation is forced to −1.0.    │
│ ICA insight: Wavelet decomposition creates virtual channels; kurtosis       │
│ identifies spiky artifact components for removal.                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The three radio button options:**

| Method | What it shows |
|---|---|
| CBSI (Anti-Correlation) | Uses HbO/HbR opposite relationship to cancel shared noise |
| ICA (Wavelet) | Decomposes signal into independent sources, removes spiky ones |
| ICA + CBSI (Full pipeline) | Both methods combined — ICA first, then CBSI |

**Reading the correlation metric:**
- Before cleaning: HbO/HbR correlation is near 0 or slightly negative
- After CBSI: correlation is forced to exactly -1.0 — perfect anti-correlation
- A bigger delta (more negative change) = more noise was removed

---

## Screen 7 — Tab 4: Model Comparison

Two side-by-side horizontal bar charts comparing all four classifiers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Classifier Comparison (5-Fold Stratified CV)                                │
│                                                                             │
│         Accuracy                          F1-Macro                         │
│  ┌──────────────────────────┐    ┌──────────────────────────┐              │
│  │ LDA              █████▌  │    │ LDA              ████▌   │              │
│  │              0.938±0.04  │    │               0.830      │              │
│  │                          │    │                          │              │
│  │ Random Forest    █████▌  │    │ Random Forest    ███     │              │
│  │              0.938±0.00  │    │               0.643      │              │
│  │                          │    │                          │              │
│  │ SVM (RBF)        █████   │    │ SVM (RBF)        ███     │              │
│  │              0.901±0.03  │    │               0.616      │              │
│  │                          │    │                          │              │
│  │ Gradient Boost   ████▌   │    │ Gradient Boost   ███     │              │
│  │              0.888±0.05  │    │               0.611      │              │
│  │                          │    │                          │              │
│  │  0        0.5       1.0  │    │  0        0.5       1.0  │              │
│  └──────────────────────────┘    └──────────────────────────┘              │
│                                                                             │
│ 💡 Why LDA performs well here: fNIRS datasets are often small (N < 200).   │
│    LDA's linear decision boundary is well-suited to this regime.           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Reading this chart:**

- Error bars on Accuracy show how consistent the model is across folds — wide bars = inconsistent
- F1-Macro penalises models that ignore minority classes
- LDA wins both metrics — simple models beat complex ones on small datasets
- Random Forest has the same accuracy as LDA but far lower F1 — it struggles on MENTAL vs MOTOR

---

## Screen 8 — Tab 5: Confusion Matrix

A 3×3 heatmap showing where the best model (Gradient Boosting) gets confused.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Confusion Matrix — Best Model (Gradient Boosting, held-out test set)        │
│                                                                             │
│                    Predicted                                                │
│              REST    MENTAL    MOTOR                                        │
│        ┌──────────┬──────────┬──────────┐                                  │
│  REST  │    11    │    0     │    1     │  ← 11 correct, 1 misread as MOTOR│
│        ├──────────┼──────────┼──────────┤                                  │
│ MENTAL │    0     │    8     │    4     │  ← 4 MENTAL mislabelled as MOTOR │
│        ├──────────┼──────────┼──────────┤                                  │
│  MOTOR │    0     │    1     │    11    │  ← mostly correct                │
│        └──────────┴──────────┴──────────┘                                  │
│                                                                             │
│  (darker green = higher count)                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**How to read it:**
- Rows = what the trial actually was (true label)
- Columns = what the model predicted
- Diagonal = correct predictions (you want these to be high)
- Off-diagonal = mistakes

**The main confusion:** MENTAL is sometimes predicted as MOTOR. This makes sense —
both are active brain states with HbO rising and HbR falling. MOTOR just has a
stronger response, so the boundary between them is not always clear.

REST is almost never confused with anything else — its flat signal is easy to identify.

---

## Screen 9 — Live Decode

Below the tabs, the sidebar's **🔍 Decode Single Epoch** button triggers a live
real-time prediction. You choose a mental state, the app generates a fresh epoch,
and the model predicts what it is.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔍 Live Decode                                                              │
│                                                                             │
│  ┌──────────────────┐   ┌───────────────────────────────────────┐          │
│  │ True class       │   │ Predicted                             │          │
│  │                  │   │                                       │          │
│  │    MENTAL        │   │    MENTAL          ✓ Correct          │          │
│  └──────────────────┘   └───────────────────────────────────────┘          │
│                                                                             │
│  Class probabilities                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  1.0 ┤                                                          │       │
│  │  0.8 ┤                   ████                                   │       │
│  │  0.6 ┤                   ████  0.71                             │       │
│  │  0.4 ┤                   ████                                   │       │
│  │  0.2 ┤  ██               ████          ██                       │       │
│  │  0.0 ┤  ██  0.11         ████          ██  0.18                 │       │
│  │      └──────────────────────────────────────────────────        │       │
│  │            REST          MENTAL         MOTOR                   │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What this section demonstrates:**
- The model runs in real time on a brand new synthetic epoch (not one it trained on)
- The probability bar chart shows confidence — a tall bar on one class = the model is sure
- Sometimes it gets it wrong — this is normal, especially for MENTAL vs MOTOR confusion
- Click **🔍 Decode Single Epoch** multiple times to see different results each time

---

## Full App Flow Summary

```
Open app
    │
    ├── Adjust sidebar controls (optional)
    │
    ▼
Click ▶ Run Pipeline
    │
    ├── Spinner 1: Generate data
    ├── Spinner 2: Preprocess
    ├── Spinner 3: Extract features
    └── Spinner 4: Train classifiers
    │
    ▼
Metrics row appears (epochs, features, accuracy)
    │
    ├── Tab 1: Raw Signal     — see raw HbO/HbR per class
    ├── Tab 2: Preprocessed   — see cleaned signal, baseline at zero
    ├── Tab 3: Artifact Removal — compare CBSI / ICA / both
    ├── Tab 4: Model Comparison — accuracy and F1 bar charts
    └── Tab 5: Confusion Matrix — see where the model makes mistakes
    │
    ▼
Pick a mental state in sidebar → Click 🔍 Decode Single Epoch
    │
    └── Live prediction with probability bar chart
```

---

*Nancy Tyagi — May 2026*
