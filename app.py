"""
fNIRS Mental State Decoder — Interactive Demo
Nancy Tyagi | May 2026

Run with: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from data.generate import generate_dataset, FNIRSConfig
from utils.preprocess import preprocess_dataset, PreprocConfig
from utils.features import build_feature_matrix, extract_features
from models.classify import evaluate_all_models, train_best_model, CLASS_NAMES

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="fNIRS Mental State Decoder",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 fNIRS Mental State Decoder")
st.caption(
    "A prototype BCI pipeline decoding cognitive state from functional Near-Infrared "
    "Spectroscopy signals."
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Config")
    n_epochs = st.slider("Epochs per class", 30, 100, 60, step=10)
    n_channels = st.select_slider("Channels", options=[4, 8, 16, 32], value=16)
    fs = st.select_slider("Sampling rate (Hz)", options=[5.0, 10.0, 20.0], value=10.0)
    epoch_len = st.slider("Epoch length (s)", 5, 15, 10)
    artifact_thresh = st.slider("Artifact rejection threshold (σ)", 2.0, 5.0, 3.0, step=0.5)
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Live Decode**")
    live_class = st.selectbox("Simulate mental state", CLASS_NAMES, index=0)
    decode_btn = st.button("🔍 Decode Single Epoch", use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "model" not in st.session_state:
    st.session_state.model = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Generating synthetic fNIRS data..."):
        cfg = FNIRSConfig(
            fs=fs, n_channels=n_channels,
            epoch_len=epoch_len, n_epochs_per_class=n_epochs
        )
        raw = generate_dataset(cfg)

    with st.spinner("Preprocessing (bandpass filter + baseline correction + artifact rejection)..."):
        proc_cfg = PreprocConfig(fs=fs, artifact_threshold=artifact_thresh)
        proc = preprocess_dataset(raw, proc_cfg)

    with st.spinner("Extracting features..."):
        X, y = build_feature_matrix(proc)

    with st.spinner("Training & evaluating classifiers..."):
        cv_results = evaluate_all_models(X, y, n_splits=5)
        model, cm, report, acc = train_best_model(X, y)

    st.session_state.results = {
        "raw": raw, "proc": proc, "X": X, "y": y,
        "cv": cv_results, "cm": cm, "acc": acc,
    }
    st.session_state.model = model
    st.session_state.dataset = proc

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.results is not None:
    res = st.session_state.results
    proc = res["proc"]
    raw = res["raw"]

    # ── Metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    total = len(raw["y"])
    kept = len(proc["y"])
    col1.metric("Epochs generated", total)
    col2.metric("After artifact rejection", kept, delta=f"-{total - kept} removed")
    col3.metric("Feature dimensions", res["X"].shape[1])
    col4.metric("Best model accuracy", f"{res['acc']:.1%}")

    st.divider()

    # ── Tab layout ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📡 Raw Signal", "🔬 Preprocessed", "🧹 Artifact Removal", "📊 Model Comparison", "🗺️ Confusion Matrix"]
    )

    # Tab 1: Raw signal viz
    with tab1:
        st.subheader("Raw fNIRS Signal — Example Epochs")
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        fig.patch.set_facecolor("#0e1117")
        colors = {"HbO": "#e74c3c", "HbR": "#3498db"}

        for i, label in enumerate([0, 1, 2]):
            ax = axes[i]
            ax.set_facecolor("#1a1d27")
            idx = np.where(raw["y"] == label)[0][0]
            t = np.linspace(0, raw["cfg"].epoch_len, raw["X_hbo"].shape[-1])
            # Show first 3 channels
            for ch in range(min(3, raw["X_hbo"].shape[1])):
                alpha = 1.0 - ch * 0.2
                ax.plot(t, raw["X_hbo"][idx, ch], color="#e74c3c", alpha=alpha,
                        linewidth=1.2, label=f"HbO ch{ch}" if ch == 0 else "")
                ax.plot(t, raw["X_hbr"][idx, ch], color="#3498db", alpha=alpha,
                        linewidth=1.2, label=f"HbR ch{ch}" if ch == 0 else "")
            ax.set_ylabel(CLASS_NAMES[label], color="white", fontsize=11)
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            if i == 0:
                ax.legend(loc="upper right", facecolor="#1a1d27", labelcolor="white")

        axes[-1].set_xlabel("Time (s)", color="white")
        fig.suptitle("Raw HbO (red) and HbR (blue) signals per class", color="white", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tab 2: Preprocessed
    with tab2:
        st.subheader("After Preprocessing (bandpass + baseline correction)")
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        fig.patch.set_facecolor("#0e1117")

        for i, label in enumerate([0, 1, 2]):
            ax = axes[i]
            ax.set_facecolor("#1a1d27")
            idxs = np.where(proc["y"] == label)[0]
            if len(idxs) == 0:
                ax.text(0.5, 0.5, "No epochs", ha="center", color="white", transform=ax.transAxes)
                continue
            idx = idxs[0]
            t = np.linspace(0, proc["cfg"].epoch_len, proc["X_hbo"].shape[-1])
            for ch in range(min(3, proc["X_hbo"].shape[1])):
                alpha = 1.0 - ch * 0.2
                ax.plot(t, proc["X_hbo"][idx, ch], color="#e74c3c", alpha=alpha, linewidth=1.2)
                ax.plot(t, proc["X_hbr"][idx, ch], color="#3498db", alpha=alpha, linewidth=1.2)
            ax.axhline(0, color="#555", linestyle="--", linewidth=0.8)
            ax.set_ylabel(CLASS_NAMES[label], color="white", fontsize=11)
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")

        axes[-1].set_xlabel("Time (s)", color="white")
        fig.suptitle("Preprocessed signals — baseline at zero, noise reduced", color="white", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tab 3: Artifact Removal
    with tab3:
        st.subheader("Software Artifact Removal — ICA + Anti-Correlation (CBSI)")
        st.info(
            "Single-sensor wearables have no short-separation reference channel. "
            "These two algorithms remove scalp noise in software using signal structure alone — no reference channel needed."
        )

        from utils.artifact_removal import apply_cbsi, apply_ica_multichannel, full_artifact_removal_pipeline, ICAConfig

        method = st.radio("Show cleaning method:", ["CBSI (Anti-Correlation)", "ICA (Wavelet)", "ICA + CBSI (Full pipeline)"], horizontal=True)
        class_sel = st.selectbox("Mental state to visualize", ["REST", "MENTAL", "MOTOR"], index=2, key="art_class")
        label_map2 = {"REST": 0, "MENTAL": 1, "MOTOR": 2}
        lbl = label_map2[class_sel]
        idxs2 = np.where(proc["y"] == lbl)[0]
        if len(idxs2) == 0:
            st.warning("No epochs of this class survived artifact rejection.")
        else:
            ep_idx = idxs2[0]
            hbo_ep = proc["X_hbo"][ep_idx]
            hbr_ep = proc["X_hbr"][ep_idx]
            t_ep = np.linspace(0, proc["cfg"].epoch_len, hbo_ep.shape[1])

            if method == "CBSI (Anti-Correlation)":
                hbo_c, hbr_c, diag = apply_cbsi(hbo_ep, hbr_ep, return_diagnostics=True)
                corr_b = diag["mean_corr_before"]
                corr_a = diag["mean_corr_after"]
                method_label = "CBSI"
            elif method == "ICA (Wavelet)":
                hbo_c, _ = apply_ica_multichannel(hbo_ep, ICAConfig(fs=proc["cfg"].fs))
                hbr_c, _ = apply_ica_multichannel(hbr_ep, ICAConfig(fs=proc["cfg"].fs))
                corr_b = np.mean([np.corrcoef(hbo_ep[c], hbr_ep[c])[0,1] for c in range(hbo_ep.shape[0])])
                corr_a = np.mean([np.corrcoef(hbo_c[c], hbr_c[c])[0,1] for c in range(hbo_c.shape[0])])
                method_label = "ICA"
            else:
                hbo_c, hbr_c, _ = full_artifact_removal_pipeline(hbo_ep, hbr_ep, fs=proc["cfg"].fs)
                corr_b = np.mean([np.corrcoef(hbo_ep[c], hbr_ep[c])[0,1] for c in range(hbo_ep.shape[0])])
                corr_a = np.mean([np.corrcoef(hbo_c[c], hbr_c[c])[0,1] for c in range(hbo_c.shape[0])])
                method_label = "ICA + CBSI"

            m1, m2, m3 = st.columns(3)
            m1.metric("HbO-HbR corr before", f"{corr_b:.3f}")
            m2.metric("HbO-HbR corr after", f"{corr_a:.3f}", delta=f"{corr_a - corr_b:+.3f}")
            m3.metric("Method", method_label)

            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            fig.patch.set_facecolor("#0e1117")
            hbo_avg_raw = hbo_ep.mean(axis=0)
            hbr_avg_raw = hbr_ep.mean(axis=0)
            hbo_avg_c   = hbo_c.mean(axis=0)
            hbr_avg_c   = hbr_c.mean(axis=0)

            ax = axes[0]
            ax.set_facecolor("#1a1d27")
            ax.plot(t_ep, hbo_avg_raw, color="#e74c3c", linewidth=1.5, label="HbO raw")
            ax.plot(t_ep, hbr_avg_raw, color="#3498db", linewidth=1.5, label="HbR raw")
            ax.set_ylabel("Raw signal", color="white"); ax.legend(loc="upper right"); ax.grid(True); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

            ax = axes[1]
            ax.set_facecolor("#1a1d27")
            ax.plot(t_ep, hbo_avg_c, color="#f39c12", linewidth=1.5, label=f"HbO after {method_label}")
            ax.plot(t_ep, hbr_avg_c, color="#2ecc71", linewidth=1.5, label=f"HbR after {method_label}")
            ax.set_ylabel("Cleaned signal", color="white"); ax.legend(loc="upper right"); ax.grid(True); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

            ax = axes[2]
            ax.set_facecolor("#1a1d27")
            noise = hbo_avg_raw - hbo_avg_c
            ax.plot(t_ep, noise, color="#9b59b6", linewidth=1.3, label="Removed (HbO raw − cleaned)")
            ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Time (s)", color="white"); ax.set_ylabel("Noise removed", color="white")
            ax.legend(loc="upper right"); ax.grid(True); ax.tick_params(colors="white"); ax.spines[:].set_color("#444")

            fig.suptitle(f"{class_sel} epoch — {method_label} artifact removal (channel average)", color="white", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(
                "**CBSI insight:** After correction, HbO-HbR correlation is forced to −1.0. "
                "This mathematically removes all co-correlated (systemic) components. "
                "**ICA insight:** Wavelet decomposition creates virtual channels; kurtosis identifies "
                "spiky artifact components for removal. Both methods are blind — no reference channel needed."
            )

    # Tab 4: Model Comparison
    with tab4:
        st.subheader("Classifier Comparison (5-Fold Stratified CV)")
        cv = res["cv"]
        names = list(cv.keys())
        accs = [cv[n]["acc_mean"] for n in names]
        stds = [cv[n]["acc_std"] for n in names]
        f1s  = [cv[n]["f1_mean"] for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0e1117")
        bar_color = "#2ecc71"

        for ax, vals, errs, title in [
            (ax1, accs, stds, "Accuracy"),
            (ax2, f1s, None, "F1-Macro"),
        ]:
            ax.set_facecolor("#1a1d27")
            bars = ax.barh(names, vals, xerr=errs, color=bar_color, alpha=0.85,
                           error_kw={"ecolor": "white", "capsize": 4})
            ax.set_xlim(0, 1.05)
            ax.set_xlabel(title, color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            ax.set_title(title, color="white", fontsize=12)
            for bar, val in zip(bars, vals):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", color="white", fontsize=10)

        fig.suptitle("5-Fold Cross-Validation Results", color="white", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "💡 **Why LDA performs well here:** fNIRS datasets are often small (N < 200 epochs). "
            "LDA's linear decision boundary regularized by the within-class covariance is well-suited "
            "to this regime — a well-known result in the BCI literature."
        )

    # Tab 5: Confusion matrix
    with tab5:
        st.subheader("Confusion Matrix — Best Model (Gradient Boosting, held-out test set)")
        cm = res["cm"]
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1a1d27")

        cmap = LinearSegmentedColormap.from_list("fnirs", ["#1a1d27", "#2ecc71"])
        im = ax.imshow(cm, cmap=cmap)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASS_NAMES, color="white")
        ax.set_yticklabels(CLASS_NAMES, color="white")
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("True", color="white")
        ax.tick_params(colors="white")

        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Live decode ───────────────────────────────────────────────────────────────
if decode_btn and st.session_state.model is not None:
    st.divider()
    st.subheader("🔍 Live Decode")

    label_map = {"REST": 0, "MENTAL": 1, "MOTOR": 2}
    true_label = label_map[live_class]

    from data.generate import generate_epoch, FNIRSConfig
    from utils.preprocess import bandpass_filter, baseline_correct, PreprocConfig

    cfg = st.session_state.dataset["cfg"]
    rng = np.random.default_rng()
    hbo, hbr = generate_epoch(true_label, cfg, rng)

    pcfg = PreprocConfig(fs=cfg.fs)
    hbo = baseline_correct(bandpass_filter(hbo, pcfg), pcfg)
    hbr = baseline_correct(bandpass_filter(hbr, pcfg), pcfg)

    feats = extract_features(hbo, hbr, cfg.fs).reshape(1, -1)
    pred = st.session_state.model.predict(feats)[0]
    proba = None
    if hasattr(st.session_state.model.named_steps["clf"], "predict_proba"):
        proba = st.session_state.model.predict_proba(feats)[0]

    col1, col2 = st.columns(2)
    col1.metric("True class", live_class)
    col2.metric("Predicted", CLASS_NAMES[pred],
                delta="✓ Correct" if pred == true_label else "✗ Incorrect")

    if proba is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1a1d27")
        colors = ["#95a5a6", "#e67e22", "#2ecc71"]
        bars = ax.bar(CLASS_NAMES, proba, color=colors, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.set_title("Class probabilities", color="white")
        for bar, p in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02,
                    f"{p:.2f}", ha="center", color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

elif decode_btn and st.session_state.model is None:
    st.warning("Run the pipeline first to train the model.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Tech stack:** Python · MNE · scikit-learn · scipy · matplotlib · Streamlit  |  "
    "**Signal:** Synthetic fNIRS with realistic HRF, Mayer waves, respiratory artifacts  |  "
    "**Decoding:** Statistical + spectral + CBSI features → LDA / RF / SVM / GradBoost"
)
