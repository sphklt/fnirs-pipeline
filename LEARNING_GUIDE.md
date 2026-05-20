# fNIRS Pipeline — Learning Guide

A beginner-friendly walkthrough of every step in this pipeline, from brain biology to
classified mental state. No prior neuroscience or signal processing knowledge required.

---

## Table of Contents

1. [What is fNIRS?](#step-1--what-is-fnirs)
2. [What does the data look like?](#step-2--what-does-the-data-look-like)
3. [Synthetic data generation](#step-3--synthetic-data-generation)
4. [Preprocessing](#step-4--preprocessing)
5. [Feature extraction](#step-5--feature-extraction)
6. [Classification](#step-6--classification)
7. [Artifact removal](#step-7--artifact-removal)

---

## The Big Picture

Before diving in, here is the full pipeline in one diagram:

```
Raw fNIRS signal (brain + noise mixed together)
    ↓  preprocess.py
Cleaned signal (noise removed, bad trials dropped)
    ↓  features.py
Feature matrix (each trial compressed into ~248 numbers)
    ↓  classify.py
Predicted mental state: REST / MENTAL / MOTOR
```

Everything below explains what happens at each arrow and why.

---

## Step 1 — What is fNIRS?

### The core idea: blood flow reveals brain activity

When a region of your brain becomes active — say you start doing mental arithmetic —
neurons in that region fire. Firing neurons consume oxygen. Your body responds by sending
more oxygenated blood to that area.

**fNIRS (functional Near-Infrared Spectroscopy)** measures this blood flow response by
shining near-infrared light through your skull. The key insight: oxygenated and
deoxygenated hemoglobin absorb light differently. By measuring how much light comes
back out, you can infer the concentration of each.

During brain activation:
- **HbO (oxygenated hemoglobin) goes up** — more oxygenated blood arriving
- **HbR (deoxygenated hemoglobin) goes down** — oxygen is being consumed, HbR is washed out

This opposite movement is called **neurovascular coupling** — neural activity causes a
vascular (blood flow) response.

### The hemodynamic response — the shape of the signal

The brain's blood flow response is not instant. It has a characteristic delayed shape
called the **Hemodynamic Response Function (HRF)**:

```
HbO amplitude
    |          ▲ peak ~5–8 seconds after stimulus
    |         / \
    |        /   \
    |       /     \___
    |______/           \_____
    |
    +----------------------------> time (seconds)
         0   5   10   15   20
```

This delay exists because the vascular system takes a few seconds to respond.
Every fNIRS analysis is built around detecting this delayed shape.

### The three mental states this pipeline decodes

| Class | Label | Description | HbO | HbR |
|---|---|---|---|---|
| REST | 0 | No task, baseline | ~0 | ~0 |
| MENTAL | 1 | Mental arithmetic, cognitive load | ↑ moderate | ↓ moderate |
| MOTOR | 2 | Finger tapping, motor movement | ↑ strong | ↓ strong |

### Where this lives in the code

`data/generate.py` — `_hemodynamic_response()` models this shape using a gamma
distribution, which matches the canonical HRF from neuroscience literature.

---

## Step 2 — What Does the Data Look Like?

### Three building blocks: samples, channels, epochs

**Sample** — one measurement at one moment in time.
The device records 10 times per second (10 Hz sampling rate). Over a 10-second trial,
that is 100 samples.

**Channel** — one sensor on the head.
The pipeline uses 16 sensors placed at different locations on the skull. Each sensor
measures HbO and HbR independently.

**Epoch** — one complete trial from start to finish.
The dataset has 60 trials per class × 3 classes = 180 epochs total.

### The spreadsheet analogy

One trial looks like a spreadsheet where rows are sensors and columns are time points:

```
           0.0s   0.1s   0.2s   ...   10.0s
Sensor 1 [ 0.02   0.03   0.01   ...   0.04 ]
Sensor 2 [ 0.25   0.27   0.30   ...   0.28 ]   ← active channel, shows HRF
...
Sensor 16[ 0.01   0.01   0.02   ...   0.01 ]
```

You have **two** such spreadsheets per trial — one for HbO, one for HbR.
And 180 trials stacked on top of each other.

### The array shapes

```
X_hbo : (180 epochs, 16 channels, 100 samples)
X_hbr : (180 epochs, 16 channels, 100 samples)
y     : (180,)   →  [0, 0, 1, 2, 1, ...]  — true labels
```

The whole job of the pipeline: look at `X_hbo` and `X_hbr`, extract patterns,
and predict `y`.

### Where this lives in the code

`data/generate.py` — `generate_dataset()` builds and returns these arrays.

---

## Step 3 — Synthetic Data Generation

### Why synthetic data?

Real fNIRS recordings require a lab, a headset, and participants doing tasks.
This pipeline was built quickly for an interview demo, so it simulates realistic
data instead — using the known physics of how fNIRS signals behave.

Think of it like generating realistic fake weather data to build and test a
weather predictor. The simulation is not real, but realistic enough to validate
the full pipeline end-to-end.

### The recipe for one channel's signal

Every channel is built from **four ingredients mixed together**:

```python
hbo[ch] = (
    brain_signal      # ingredient 1: the actual HRF you want to detect
  + mayer_wave        # ingredient 2: blood pressure oscillation noise
  + respiratory       # ingredient 3: breathing noise
  + random_noise      # ingredient 4: sensor measurement noise
)
```

#### Ingredient 1 — Brain signal (HRF)

The hemodynamic response from Step 1. Two factors scale how strongly it appears:

- **`ch_scale`** — each sensor reads slightly differently (0.7×–1.3× of average).
  Realistic — no two sensors are identical.
- **`task_factor`** — half the sensors sit over active brain regions (factor = 1.0),
  the rest over quiet regions (factor ≈ 0). Only sensors over the active region
  see the full response.

#### Ingredient 2 — Mayer waves

Blood pressure oscillates slowly at ~0.1 Hz (one cycle every 10 seconds), driven by
your autonomic nervous system. fNIRS sensors pick this up as a slow rhythmic wobble —
it has nothing to do with brain activity, but it is always present. This is noise
you need to remove later.

#### Ingredient 3 — Breathing noise

Breathing causes blood flow to shift throughout your body at ~0.3 Hz (~18 breaths
per minute). fNIRS sensors pick this up too. Same structure as Mayer waves, just
faster and slightly smaller.

#### Ingredient 4 — Random noise

Pure electronic and measurement noise — the sensor is not perfect. Modelled as
random numbers centred at zero. MOTOR trials have the most noise because head
movement adds measurement error.

### What one channel looks like

```
Signal = [slow HRF hill] + [slow 0.1Hz wobble] + [faster 0.3Hz wobble] + [jitter]
           ↑ brain signal     ↑ blood pressure      ↑ breathing              ↑ noise
```

The brain signal is real information. Everything else is noise riding on top.
The pipeline's job is to strip away the noise and keep only the brain signal.

### Where this lives in the code

`data/generate.py` — `generate_epoch()` builds one trial.
`data/generate.py` — `generate_dataset()` runs it 180 times and stacks the results.

---

## Step 4 — Preprocessing

Preprocessing is the cleaning step. Three sub-steps are applied to every epoch in order.

### Sub-step 1: Bandpass Filter

#### The idea

Your signal contains many frequencies mixed together. A filter is a dial that lets
through only a specific range of frequencies and blocks everything else.

```
blocked    |   let through    |   blocked
───────────┼──────────────────┼──────────
0 Hz     0.01 Hz           0.5 Hz       ∞
(drift)  (← brain band →)              (noise)
```

- Below 0.01 Hz: very slow baseline drift — the signal slowly wandering over minutes.
  Not brain activity.
- Above 0.5 Hz: fast electrical noise, heartbeat (~1 Hz). Not brain activity.
- Between 0.01–0.5 Hz: where fNIRS brain signals actually live.

A **bandpass filter** lets through only the middle band.

#### In the code

`utils/preprocess.py` — `bandpass_filter()`

```python
sos = butter(4, [low, high], btype='bandpass', output='sos')
return sosfiltfilt(sos, signal, axis=-1)
```

- `butter(4, ...)` — Butterworth filter, order 4. Higher order = sharper cutoff.
- `sosfiltfilt` — applies the filter forward then backward in time, cancelling any
  timing shift. The cleaned signal stays aligned with the original.

Note: Mayer waves (0.1 Hz) and breathing (0.3 Hz) survive this filter. They sit
inside the brain band. More targeted removal happens in Step 7.

---

### Sub-step 2: Baseline Correction

#### The idea

After filtering, different channels still sit at different absolute levels due to
sensor placement, skin thickness, hair, etc. We do not care about absolute levels —
we care about *change from the start of the trial*.

Baseline correction subtracts each channel's average value during the first 2 seconds
(before the task begins), so every channel starts at zero.

```
Before:  channel 3 → [0.15, 0.17, 0.22, ...]   channel 7 → [0.42, 0.44, 0.47, ...]
After:   channel 3 → [0.00, 0.02, 0.07, ...]   channel 7 → [0.00, 0.02, 0.05, ...]
```

Now all channels speak the same language: *how much did blood oxygen change from the start?*

#### In the code

`utils/preprocess.py` — `baseline_correct()`

```python
n_baseline = int(2.0 * fs)                               # first 2 seconds = 20 samples
baseline_mean = signal[..., :n_baseline].mean(axis=-1)   # one average per channel
return signal - baseline_mean                            # subtract from the whole epoch
```

---

### Sub-step 3: Artifact Rejection

#### The idea

Sometimes a trial is broken — the person sneezed, coughed, or moved their head,
causing a massive spike that has nothing to do with brain activity. Learning from
these trials would confuse the classifier.

The check: does any channel contain a value that is more than 3 standard deviations
away from the epoch mean? If yes, throw the entire epoch away.

```
Normal variation → z-score of 1–2 → fine, keep it
Motion spike     → z-score of 5+  → flag as artifact, discard
```

#### In the code

`utils/preprocess.py` — `is_artifact()`

```python
z = np.abs(sig - sig.mean()) / sig.std()   # z-score for every value
if z.max() > 3.0:
    return True                             # discard this epoch
```

### What comes out of preprocessing

A cleaned dataset — fewer epochs (some rejected), but the remaining ones have:
- No slow drift, no high-frequency noise
- All channels zeroed to their baseline
- No catastrophically corrupted trials

---

## Step 5 — Feature Extraction

### The problem feature extraction solves

After preprocessing, each epoch is still `(16 channels, 100 samples)` — 1,600 numbers
for HbO alone. A classifier cannot learn reliably from 3,200 raw numbers when you only
have 180 trials.

Feature extraction **compresses** each epoch into a small set of meaningful measurements
— capturing the important patterns and discarding redundancy.

The output is one flat list of numbers per epoch called a **feature vector**.
The classifier then learns from these vectors.

### Feature family 1: Statistical features

Five summary statistics computed per channel, capturing the shape and behaviour of the
signal over the 10-second window.

| Statistic | What it tells you |
|---|---|
| Mean | Was blood oxygen high or low on average? |
| Variance | Did the signal stay flat or fluctuate a lot? |
| Skew | Did the signal spend more time above or below its average? |
| Kurtosis | Were there sudden sharp spikes? |
| Slope | Is blood oxygen still rising at the end, or already falling? |

Computed separately for HbO and HbR across all 16 channels:
→ **160 features** (16 channels × 5 statistics × 2 chromophores)

`utils/features.py` — `_channel_stats()`

---

### Feature family 2: Spectral power

Any signal can be decomposed into waves at different frequencies. Spectral power
measures how much energy is present in specific frequency bands.

Three physiologically meaningful bands are used:

| Band | Frequency | Source |
|---|---|---|
| VLF (very low freq) | 0.01–0.04 Hz | Slow vasomotion — blood vessel wall contractions |
| Mayer wave | 0.07–0.13 Hz | Blood pressure oscillation |
| Respiratory | 0.15–0.40 Hz | Breathing |

The relative balance of power across these bands shifts between mental states — for
example, cognitive load changes heart rate variability, which shifts Mayer wave power.

→ **48 features** (16 channels × 3 bands)

`utils/features.py` — `_spectral_power()`

---

### Feature family 3: HbO-HbR coupling

The most clever feature family. During genuine brain activation, HbO goes up and HbR
goes down — they move in opposite directions (anti-correlation). During rest with no
activation, they drift independently — correlation near zero.

Two coupling measures are computed per channel:

- **Correlation** — how strongly do HbO and HbR move in opposite directions?
  REST → near 0. MENTAL/MOTOR → negative.
- **CBSI ratio** — ratio of HbO to HbR variability. Shifts away from 1.0 during tasks.

→ **32 features** (16 channels × 2 measures)

`utils/features.py` — `_hbo_hbr_coupling()`

---

### Feature family 4: Spatial asymmetry

Different tasks activate different brain hemispheres differently. Mental arithmetic
tends to activate the left prefrontal cortex more. Motor tasks activate the
contralateral motor cortex (right-hand movement → left hemisphere).

The difference between left-side and right-side channel averages captures this.

→ **8 features** (8 left/right pairs)

`utils/features.py` — `_spatial_asymmetry()`

---

### What comes out

```
One epoch: (16 channels, 100 samples)
    ↓ extract_features()
One feature vector: 248 numbers

All 180 epochs:
    X shape: (180, 248)   ← one row per trial, one column per feature
    y shape: (180,)       ← labels
```

This matrix `X` is what the classifier receives.

---

## Step 6 — Classification

### What a classifier does

Given the feature matrix `X` (180 rows × 248 columns) and labels `y`, a classifier
learns the relationship between features and mental state. After training, it can
look at a new trial's 248 features and predict which class it belongs to.

### Step inside classification: scaling

Before any classifier runs, features are scaled using `StandardScaler`.

Your 248 features are in very different numerical ranges — means near 0.01, kurtosis
values near 5.0, slopes near 0.001. Without scaling, the classifier would pay too much
attention to large-valued features just because their numbers are bigger — not because
they are more informative.

`StandardScaler` subtracts the mean and divides by the standard deviation for each
feature. Everything ends up in the same range (roughly -3 to +3).

`models/classify.py` — every model is wrapped in a `Pipeline([scaler, classifier])`.

---

### The four classifiers

#### LDA — Linear Discriminant Analysis

Finds a straight-line boundary that separates the three classes with maximum separation.
Simple, fast, and well-suited to small datasets.

**Why it wins here:** fNIRS datasets are small (180 trials). Simple models with few
parameters do not overfit small datasets. This is a well-known result in the BCI
literature — linear classifiers routinely outperform complex ones on neuroimaging data.

Result: **93.8% accuracy** — best of all four.

---

#### Random Forest

Builds 100 decision trees, each trained on a random subset of data and features.
Final prediction = majority vote across all trees.

Each tree is a flowchart: *"if feature 12 > 0.3 AND feature 47 < 0.1, predict MOTOR."*
The ensemble of trees reduces the chance any single tree is wrong.

Can capture non-linear patterns but is prone to memorising small datasets.

Result: 93.8% accuracy but low F1 — good at the dominant class, struggles on others.

---

#### SVM — Support Vector Machine

Finds the decision boundary with maximum gap (margin) to the nearest training points.
The RBF kernel allows curved, non-linear boundaries.

A classic strong baseline for small neuroimaging datasets.

Result: 90.1% accuracy.

---

#### Gradient Boosting

Builds trees one at a time, each focusing specifically on the errors made by the
previous trees. Over 100 rounds, it progressively patches its own mistakes.

Often the best performer on structured data in general, but needs more training
examples to shine. With only 180 samples, it overfits.

Result: 88.8% accuracy — worst of the four despite being the most complex.

---

### Honest evaluation: cross-validation

Simply training and testing on the same 180 trials would give near-100% accuracy —
the model just memorises the answers. That tells you nothing about real performance.

**5-fold cross-validation** solves this:

```
All 180 trials split into 5 equal groups:

Round 1: train on groups 2+3+4+5  →  test on group 1
Round 2: train on groups 1+3+4+5  →  test on group 2
Round 3: train on groups 1+2+4+5  →  test on group 3
Round 4: train on groups 1+2+3+5  →  test on group 4
Round 5: train on groups 1+2+3+4  →  test on group 5
```

Each trial is tested exactly once, on a model that never saw it during training.
Final accuracy = average across all 5 rounds. This is an honest estimate of
how the model would perform on new, unseen data.

### Reading the results

**Accuracy** — what fraction of all predictions were correct.

**F1-Macro** — balances precision (when you predict MENTAL, how often are you right?)
and recall (of all real MENTAL trials, how many did you catch?). Macro means each class
is weighted equally. A large gap between accuracy and F1 means the model is
good at one class but poor at others.

`models/classify.py` — `evaluate_all_models()` runs cross-validation.
`models/classify.py` — `train_best_model()` trains on 80%, tests on 20%, produces a confusion matrix.

---

## Step 7 — Artifact Removal (Advanced Cleaning)

### Why preprocessing alone is not enough

Preprocessing (Step 4) removes slow drift, high-frequency noise, and the worst broken
trials. But one category of noise survives: **systemic noise** — body-wide blood flow
changes that affect the scalp directly above the sensor.

Examples:
- A sudden stress response → blood flow increases everywhere, including the scalp
- Blood pressure fluctuations → picked up by every sensor
- Scalp blood flow changes → fNIRS sensors sit on the scalp, not the brain

These mimic real brain signals. They are in the right frequency range, right amplitude,
and pass artifact rejection. This is the hard problem — and Temple's specific engineering
challenge, since their single sensor sits on the scalp with no reference channel to
subtract against.

Two methods address this: **CBSI** and **ICA**.

---

### Method 1: CBSI (Correlation-Based Signal Improvement)

#### The core insight

| Source | HbO | HbR | Relationship |
|---|---|---|---|
| Real brain activation | ↑ | ↓ | Opposite (anti-correlated) |
| Systemic noise | ↑ | ↑ | Same direction (co-correlated) |

When noise hits, both signals go up together. When the brain activates, they move
oppositely. CBSI exploits this difference to separate brain signal from noise.

#### The maths, simply explained

```
alpha = std(HbO) / std(HbR)        ← how much bigger is HbO's swing vs HbR's?

HbO_clean = 0.5 × (HbO - alpha × HbR)
HbR_clean = 0.5 × (HbR - (1/alpha) × HbO)
```

Work through two scenarios:

```
Pure noise (both go up together):
  HbO = +0.10,  HbR = +0.10,  alpha ≈ 1.0
  HbO_clean = 0.5 × (0.10 - 1.0 × 0.10) = 0.0   ← noise cancelled

Real brain signal (HbO up, HbR down):
  HbO = +0.10,  HbR = -0.10,  alpha ≈ 1.0
  HbO_clean = 0.5 × (0.10 - 1.0 × (-0.10)) = 0.10  ← signal preserved
```

Noise cancels itself out. Brain signal survives. No reference channel needed.

After CBSI, the HbO/HbR correlation is pushed toward -1.0 — everything remaining
obeys the neurovascular coupling constraint.

`utils/artifact_removal.py` — `apply_cbsi()`

---

### Method 2: ICA (Independent Component Analysis)

#### The cocktail party analogy

Imagine 3 microphones in a room where 3 people are talking simultaneously. Each
microphone picks up all three voices mixed together at different volumes. ICA is the
algorithm that, given only the microphone recordings, **unmixes** them back into
individual voices.

In fNIRS: the "voices" are independent signal sources (brain activity, heartbeat,
Mayer waves, motion artifact). ICA finds the unmixed sources. You then identify which
ones are noise and discard them before reconstructing the signal.

#### The problem for a single-sensor device

Standard ICA needs multiple channels (multiple "microphones") to work. A
single-sensor wearable has one channel. You cannot unmix from a single recording.

**Solution: wavelet decomposition creates virtual channels.**

A wavelet transform decomposes one signal into several components, each capturing
behaviour at a different frequency band:

```
One channel (100 samples)
    ↓ wavelet decomposition
Virtual channel 1: very fast fluctuations (2.5–5 Hz)
Virtual channel 2: medium-fast fluctuations (1.25–2.5 Hz)
Virtual channel 3: medium fluctuations (0.6–1.25 Hz)
Virtual channel 4: slow fluctuations (0–0.6 Hz)
```

These four virtual channels are fed into ICA as if they were four separate sensors.

#### Identifying artifact components by kurtosis

After ICA unmixes the virtual channels into independent components, each component
is examined using **kurtosis** — a measure of how spiky a signal is.

- Brain hemodynamic signals: slow, smooth → low kurtosis
- Motion artifacts: sudden large spikes → high kurtosis
- Mayer waves: very regular sine wave → negative kurtosis

Components outside the normal kurtosis range are flagged as artifacts and zeroed out.
The clean signal is reconstructed from only the remaining components.

`utils/artifact_removal.py` — `apply_ica_single_channel()`, `apply_ica_multichannel()`

---

### The combined pipeline: ICA then CBSI

```python
# Step 1: ICA — removes spiky broadband artifacts (motion, cardiac)
hbo_ica = apply_ica_multichannel(hbo)

# Step 2: CBSI — removes systemic hemodynamics using HbO/HbR anti-correlation
hbo_clean, hbr_clean = apply_cbsi(hbo_ica, hbr_ica)
```

Order matters. ICA first removes irregular spiky artifacts. Then CBSI removes the
smoother systemic noise. Doing it the other way would let spikes corrupt the CBSI
subtraction.

`utils/artifact_removal.py` — `full_artifact_removal_pipeline()`

---

### CBSI vs ICA at a glance

| | CBSI | ICA |
|---|---|---|
| Removes | Systemic noise (HbO + HbR move together) | Spiky artifacts, motion, cardiac |
| How | Subtracts the co-correlated component | Unmixes signal, discards artifact sources |
| Needs | Both HbO and HbR | Single channel (with wavelet trick) |
| Speed | Very fast | Slower — iterative algorithm |
| Limitation | Cannot remove noise affecting only one signal | Needs enough samples to converge |

---

## The Full Pipeline End to End

```
Raw fNIRS signal
  180 epochs × 16 channels × 100 samples
  (brain signal + Mayer waves + breathing + noise)
        │
        ▼  data/generate.py
  Synthetic dataset generated with realistic physiology
        │
        ▼  utils/preprocess.py
  Bandpass filter  →  keeps only 0.01–0.5 Hz
  Baseline correct →  every channel starts at zero
  Artifact reject  →  broken trials discarded
        │
        ▼  utils/artifact_removal.py  (optional deeper cleaning)
  ICA              →  removes spiky broadband artifacts
  CBSI             →  removes systemic scalp hemodynamics
        │
        ▼  utils/features.py
  248 features per epoch:
    statistical (160) + spectral (48) + coupling (32) + spatial (8)
        │
        ▼  models/classify.py
  StandardScaler → normalise all features
  LDA / RF / SVM / Gradient Boosting → learn from features
  5-fold cross-validation → honest accuracy estimate
        │
        ▼
  Predicted mental state: REST / MENTAL / MOTOR
```

---

## Key Terms Reference

| Term | Meaning |
|---|---|
| fNIRS | Functional Near-Infrared Spectroscopy — brain imaging via light |
| HbO | Oxygenated hemoglobin — goes up during brain activation |
| HbR | Deoxygenated hemoglobin — goes down during brain activation |
| HRF | Hemodynamic Response Function — the characteristic slow rise-and-fall shape |
| Neurovascular coupling | The link between neural firing and blood flow changes |
| Epoch | One trial — a single recording window (e.g. 10 seconds) |
| Channel | One sensor on the skull |
| Sample | One measurement at one point in time |
| Bandpass filter | Lets through only a chosen frequency range, blocks the rest |
| Baseline correction | Subtracts the pre-task average so all channels start at zero |
| Artifact rejection | Discards trials with unreasonably large spikes |
| Feature vector | A compact set of numbers summarising one epoch |
| CBSI | Correlation-Based Signal Improvement — removes systemic noise using HbO/HbR anti-correlation |
| ICA | Independent Component Analysis — unmixes a signal into sources, removes artifact sources |
| Kurtosis | How spiky a signal is — high kurtosis flags motion artifacts |
| Cross-validation | Evaluating a model on data it never trained on — gives an honest accuracy estimate |
| F1-Macro | Accuracy metric that weights all classes equally, penalises class imbalance |

---

*Nancy Tyagi — built as a Temple interview demo, May 2026*
