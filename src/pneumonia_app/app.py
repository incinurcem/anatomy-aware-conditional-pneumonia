"""Pneumonia AI Detection — Clinical Dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from core.config import DEFAULTS, EXPERIMENTS, EXP_LABELS, SEEDS, PIPELINES
from core.data import gather_runs, aggregate_seed, ablation_pivot, load_csv
from core.stats import all_ttests
from core.plots import (
    plot_roc_overlay, plot_pr_overlay, plot_delta_bars,
    plot_confusion_grid, plot_metric_bars,
)
from core.tables import display_metrics_table, display_ablation_table, display_ttest_table
from core.input_prep import prepare_input
from core.models import load_model, parse_experiment, get_target_layer
from core.gradcam import GradCAM, overlay_heatmap
from core.clinical import (
    load_image_from_upload, otsu_lung_mask, extract_roi,
    soft_mask_apply, make_overlay, preprocess_for_model,
    predict_proba, severity_label, compute_conditional_features,
)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Pneumonia AI Detection",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# RICH CSS THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1400px; }

/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(135deg, #0F2027 0%, #2C5364 50%, #2A9D8F 100%);
    padding: 2.5rem 2.2rem;
    border-radius: 18px;
    color: white;
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(244,162,97,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    color: white !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.4rem 0 !important;
    letter-spacing: -0.8px;
    position: relative;
}
.hero p {
    color: rgba(255,255,255,0.88);
    font-size: 1.1rem;
    margin: 0;
    position: relative;
}
.hero-stats { display: flex; gap: 1rem; margin-top: 1.6rem; flex-wrap: wrap; position: relative; }
.hero-stat {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    padding: 0.85rem 1.4rem;
    border-radius: 10px;
    backdrop-filter: blur(12px);
    flex: 1; min-width: 180px;
}
.hero-stat-num { font-size: 1.7rem; font-weight: 800; color: #fff; line-height: 1; }
.hero-stat-label { font-size: 0.82rem; color: rgba(255,255,255,0.82); margin-top: 4px; }

/* ---------- DISCLAIMER ---------- */
.disclaimer {
    background: linear-gradient(135deg, #FEF3C7 0%, #FED7AA 100%);
    border-left: 4px solid #F59E0B;
    padding: 1rem 1.4rem;
    border-radius: 10px;
    color: #78350F;
    margin: 0 0 1.5rem;
    font-size: 0.92rem;
    box-shadow: 0 2px 8px rgba(245,158,11,0.15);
}

/* ---------- SECTION HEADER ---------- */
.sec-h {
    font-size: 1.25rem;
    font-weight: 700;
    color: #111827;
    margin: 1.6rem 0 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #2A9D8F;
    display: inline-block;
}

/* ---------- EMPTY STATE ---------- */
.empty-state {
    background: linear-gradient(135deg, #EBF5F9 0%, #DBEAFE 100%);
    border-radius: 18px;
    padding: 4.5rem 2rem;
    text-align: center;
    border: 2px dashed #38BDF8;
    margin-top: 1rem;
}
.empty-icon { font-size: 4.5rem; margin-bottom: 1rem; }
.empty-title { font-size: 1.6rem; font-weight: 700; color: #075985; margin: 0 0 0.6rem; }
.empty-sub { color: #0369A1; margin: 0; font-size: 1rem; line-height: 1.6; }

/* ---------- RESULT CARD ---------- */
.result-card {
    background: white;
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    border: 3px solid;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.result-positive { border-color: #DC2626; background: linear-gradient(135deg, #FFFFFF 0%, #FEF2F2 100%); }
.result-negative { border-color: #16A34A; background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%); }

.r-label {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: #6B7280;
}
.r-pred {
    font-size: 3rem; font-weight: 800; margin: 0.4rem 0;
    letter-spacing: -1.5px; line-height: 1;
}
.r-pred.pos { color: #DC2626; }
.r-pred.neg { color: #16A34A; }

.r-prob-label { font-size: 0.85rem; color: #6B7280; margin: 1rem 0 0.2rem; font-weight: 500; }
.r-prob {
    font-size: 4rem; font-weight: 800; color: #111827;
    line-height: 1; letter-spacing: -2px;
}

/* ---------- SEVERITY BADGE ---------- */
.sev-badge {
    display: inline-block; padding: 0.5rem 1.1rem;
    border-radius: 24px; font-weight: 700; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.8px; color: white;
    margin: 0.6rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

/* ---------- GAUGE ---------- */
.gauge {
    background: #F3F4F6; border-radius: 100px; height: 26px;
    position: relative; overflow: visible; margin: 1.2rem 0 0.4rem;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
}
.gauge-fill {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #16A34A 0%, #84CC16 30%, #FBBF24 55%, #F97316 75%, #DC2626 100%);
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.gauge-marker {
    position: absolute; top: -5px;
    width: 4px; height: 36px;
    background: #1F2937; border-radius: 2px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
}
.gauge-marker::after {
    content: '▼'; position: absolute; top: -16px; left: -6px;
    font-size: 12px; color: #1F2937;
}
.gauge-labels {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: #6B7280; margin-top: 0.4rem; font-weight: 500;
}

/* ---------- META INFO ---------- */
.meta-info {
    background: #F9FAFB; padding: 1rem 1.2rem; border-radius: 10px;
    font-size: 0.86rem; color: #4B5563; margin-top: 1.2rem; line-height: 1.7;
    border: 1px solid #E5E7EB;
}
.meta-info strong { color: #111827; font-weight: 600; }

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: #F3F4F6; padding: 0.4rem;
    border-radius: 12px; margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 1.6rem; border-radius: 8px;
    font-weight: 600; color: #4B5563; font-size: 0.95rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2A9D8F, #2E86AB) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(46,134,171,0.35);
}

/* ---------- IMAGE LABELS ---------- */
[data-testid="stImage"] {
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}

/* ---------- CARD INFO ---------- */
.info-card {
    background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
    padding: 1.2rem; border-radius: 12px;
    font-size: 0.92rem; color: #374151; line-height: 1.6;
    border-left: 4px solid #2A9D8F;
}

h1, h2, h3 { color: #111827; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER — robust column detection for prediction CSVs
# ============================================================
def _detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

LABEL_COLS = ["label", "y_true", "true_label", "target", "y", "true"]
PROB_COLS  = ["prob", "y_prob", "probability", "score", "pred_prob", "pred"]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🩻 Pneumonia AI")
    st.caption("Chest X-Ray Analysis Platform")
    st.divider()

    st.markdown("### 📁 Data Sources")
    nn_root = st.text_input("nnU-Net root", DEFAULTS["nnUNet_root"])
    ot_root = st.text_input("Otsu root", DEFAULTS["Otsu_root"])

    with st.expander("🔧 Advanced"):
        threshold_mode = st.radio("Threshold mode", ["youden", "0.5"], horizontal=True)
        normalize_mode = st.selectbox("Grad-CAM normalize", ["imagenet", "none"])
        nn_csv = st.text_input("nnU-Net test CSV", DEFAULTS["nnUNet_test_csv"])
        ot_csv = st.text_input("Otsu test CSV", DEFAULTS["Otsu_test_csv"])

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Loading models..."):
    runs, df_long = gather_runs(nn_root, ot_root)

if df_long.empty:
    st.error("⚠️ No models found. Check sidebar paths.")
    st.stop()

agg = aggregate_seed(df_long)
p_auc, p_str = ablation_pivot(agg)
ttests = all_ttests(df_long)
total_runs = len(runs)
best = agg.sort_values("auc_mean", ascending=False).iloc[0]
n_sig = (ttests["p"] < 0.05).sum() if not ttests.empty else 0


# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="hero">
    <h1>🩻 Pneumonia AI Detection System</h1>
    <p>Deep-learning chest X-ray analysis with anatomy-aware classification</p>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-num">{total_runs}</div>
            <div class="hero-stat-label">Trained Models</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-num">{best['auc_mean']:.4f}</div>
            <div class="hero-stat-label">Best Test AUC</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-num">3-seed × 2 pipelines</div>
            <div class="hero-stat-label">Statistical Validation</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-num">RSNA + Kermany</div>
            <div class="hero-stat-label">Multi-Domain Tested</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN TABS — Diagnose first, research secondary
# ============================================================
tab_dx, tab_research = st.tabs([
    "🏥  Diagnose Patient",
    "📊  Research Metrics",
])


# ============================================================
# DIAGNOSE TAB — primary clinical interface
# ============================================================
with tab_dx:
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Research Demo Only — Not a Medical Device.</strong>
        This system is for research and educational purposes only. Predictions reflect the
        RSNA training distribution and must not be used for clinical decisions.
        Always consult qualified medical professionals.
    </div>
    """, unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 2.2], gap="large")

    # ---- LEFT: Upload + model selection ----
    with col_input:
        st.markdown('<div class="sec-h">📤 Upload X-Ray</div>', unsafe_allow_html=True)
        upload = st.file_uploader(
            "Drop a chest X-ray",
            type=["png", "jpg", "jpeg", "dcm", "dicom"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sec-h">⚙️ Model Selection</div>', unsafe_allow_html=True)

        available = sorted(set((pl, exp, s) for (pl, exp, s) in runs.keys()))

        sel_pl = st.selectbox(
            "Segmentation Pipeline",
            sorted(set(p for p, _, _ in available)),
            help="nnU-Net = deep CXR-pretrained · Otsu = classical thresholding",
        )

        non_cond_exps = [e for e in EXPERIMENTS if not e.startswith("conditional_")]
        cond_exps     = [e for e in EXPERIMENTS if e.startswith("conditional_")]
        all_exps_ord  = non_cond_exps + cond_exps
        exp_opts      = [e for e in all_exps_ord
                         if any(p == sel_pl and exp == e for p, exp, _ in available)]

        sel_exp = st.selectbox(
            "Input Configuration",
            exp_opts,
            format_func=lambda e: EXP_LABELS.get(e, e),
            help="Plain = full image · ROI = lung crop · Masked-ROI = mask-filtered crop. "
                 "Cond variants add 22 hand-crafted features.",
        )

        seed_opts = sorted(set(s for p, e, s in available if p == sel_pl and e == sel_exp))
        sel_seed = st.selectbox("Random Seed", seed_opts, index=0)

        compare_modes = st.checkbox(
            "🔁 Compare across all configurations",
            help="Run all 6 configurations (plain/ROI/masked-ROI × ±cond) side-by-side",
        )

    # ---- RIGHT: Result panel ----
    with col_output:
        if upload is None:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🩻</div>
                <div class="empty-title">Awaiting X-Ray Upload</div>
                <p class="empty-sub">
                    Upload a chest X-ray on the left to receive an AI-assisted analysis<br>
                    including <strong>segmentation</strong>, <strong>prediction probability</strong>,
                    <strong>severity</strong>, and <strong>Grad-CAM heatmap</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                rgb_pil, gray_pil = load_image_from_upload(upload)
            except Exception as e:
                st.error(f"Could not read image: {e}")
                st.stop()

            rgb_arr = np.array(rgb_pil)
            gray_arr = np.array(gray_pil)

            with st.spinner("🔍 Running segmentation + inference..."):
                mask = otsu_lung_mask(gray_pil)
                roi_arr, _ = extract_roi(rgb_arr, mask, padding=15)
                roi_mask, _ = extract_roi(np.stack([mask] * 3, -1), mask, padding=15)
                roi_mask = roi_mask[..., 0]
                masked_roi = soft_mask_apply(roi_arr, roi_mask, alpha=0.1)
                overlay_seg = make_overlay(rgb_arr, mask)

                # Build a grayscale ROI crop for GLCM features
                ys2, xs2 = np.where(mask > 0)
                if len(xs2) > 0:
                    pad = 15
                    x0 = max(0, xs2.min() - pad); x1 = min(gray_arr.shape[1], xs2.max() + pad)
                    y0 = max(0, ys2.min() - pad); y1 = min(gray_arr.shape[0], ys2.max() + pad)
                    roi_gray_arr = gray_arr[y0:y1, x0:x1]
                else:
                    roi_gray_arr = gray_arr

            # Pipeline preview
            st.markdown('<div class="sec-h">🔬 Preprocessing Pipeline</div>', unsafe_allow_html=True)
            p1, p2, p3, p4 = st.columns(4)
            with p1: st.image(rgb_pil,    caption="Original",   use_container_width=True)
            with p2: st.image(overlay_seg, caption="Segmentation", use_container_width=True)
            with p3: st.image(roi_arr,    caption="ROI Crop",   use_container_width=True)
            with p4: st.image(masked_roi, caption="Masked-ROI", use_container_width=True)

            # ---- Detect conditional feature column names from the selected pipeline's CSV ----
                        # ---- Detect conditional feature column names — with fallback paths ----
            def _find_csv(pl_name):
                """Try multiple known locations for the conditional CSV."""
                primary = nn_csv if pl_name == "nnUNet" else ot_csv
                if Path(primary).exists():
                    return primary
                from core.config import BASE, LOCAL
                suffix = "conditional_v3" if pl_name == "nnUNet" else "conditional_v3_otsu"
                for cand in [
                    f"{LOCAL}/{suffix}/test/test_conditional_safe.csv",
                    f"{LOCAL}/{suffix}/test/test_conditional.csv",
                    f"{BASE}/data/{suffix}/test/test_conditional_safe.csv",
                    f"{BASE}/data/{suffix}/test/test_conditional.csv",
                ]:
                    if Path(cand).exists():
                        return cand
                return primary  # original even if missing — for error message

            csv_for_pl = _find_csv(sel_pl)
            cond_feat_names = []
            cond_debug = ""

            if Path(csv_for_pl).exists():
                try:
                    _df_head = pd.read_csv(csv_for_pl, nrows=1)
                    cond_feat_names = [c for c in _df_head.columns
                                       if c.startswith(("roi_", "glcm_", "mask_"))
                                       and not c.endswith("_path")]
                    if not cond_feat_names:
                        first_cols = list(_df_head.columns)[:20]
                        cond_debug = (
                            f"CSV found at {csv_for_pl} but no `roi_*` / `glcm_*` / `mask_*` "
                            f"columns. First columns: {first_cols}"
                        )
                except Exception as e:
                    cond_debug = f"Failed to read CSV {csv_for_pl}: {e}"
            else:
                cond_debug = (
                    f"❌ CSV not found at any known location for pipeline `{sel_pl}`. "
                    f"Tried sidebar path + LOCAL + Drive fallbacks. "
                    f"Set the correct path in sidebar → Advanced."
                )

            # Show debug info when conditional configuration selected
            if sel_exp.startswith("conditional_") and cond_debug:
                st.warning(f"🔍 Conditional debug: {cond_debug}")

            # Inference helper (supports both classifier & conditional)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            def run_one(pl, exp, seed):
                run = runs.get((pl, exp, seed))
                if run is None:
                    return None
                is_cond, mode = parse_experiment(exp)

                if is_cond and not cond_feat_names:
                    return {"skip": True, "reason": cond_debug or
                            "No conditional features detected. Check CSV path in sidebar > Advanced."}

                cond_dim = len(cond_feat_names) if is_cond else 0
                try:
                    model = load_model(run["model_path"], is_cond, cond_dim, device)
                except Exception as e:
                    return {"skip": True, "reason": f"Load error: {e}"}

                if mode == "plain":   inp = rgb_arr
                elif mode == "roi":   inp = roi_arr
                else:                 inp = masked_roi

                x, pil_in = preprocess_for_model(inp)
                x = x.to(device)

                cond_tensor = None
                if is_cond:
                    csv_path = nn_csv if pl == "nnUNet" else ot_csv
                    feats = compute_conditional_features(
                        gray_arr, mask, roi_gray_arr,
                        cond_feat_names, csv_path,
                    )
                    cond_tensor = torch.from_numpy(feats).unsqueeze(0).to(device)

                with torch.no_grad():
                    if is_cond:
                        logit = model(x, cond_tensor)
                    else:
                        logit = model(x)
                    prob = torch.sigmoid(logit.squeeze()).item()

                cam_eng = GradCAM(model, get_target_layer(model, is_cond))
                cam = cam_eng(x, cond_tensor)
                cam_eng.close()

                return {
                    "prob": prob,
                    "thr": run["best_thr"],
                    "mode": mode,
                    "is_cond": is_cond,
                    "cam": overlay_heatmap(pil_in, cam),
                }

            st.markdown('<div class="sec-h">🎯 Diagnostic Result</div>', unsafe_allow_html=True)

            if compare_modes:
                # Show ALL available configurations (both classifier and conditional)
                exps_for_pl = [e for e in EXPERIMENTS
                               if any(p == sel_pl and exp == e for p, exp, _ in available)]
                # Layout: 6 cards in two rows of 3
                rows_of_cards = [exps_for_pl[:3], exps_for_pl[3:6]]
                for row in rows_of_cards:
                    if not row:
                        continue
                    cols = st.columns(len(row))
                    for i, exp in enumerate(row):
                        with cols[i]:
                            res = run_one(sel_pl, exp, sel_seed)
                            if res is None or res.get("skip"):
                                reason = res["reason"] if res else "not available"
                                st.info(f"**{EXP_LABELS[exp]}**\n\n{reason}")
                                continue
                            p = res["prob"]
                            sev_label, sev_color = severity_label(p)
                            pred = "POSITIVE" if p >= res["thr"] else "NEGATIVE"
                            cls = "pos" if pred == "POSITIVE" else "neg"
                            border = "result-positive" if pred == "POSITIVE" else "result-negative"
                            st.markdown(f"""
                            <div class="result-card {border}" style="padding: 1.2rem;">
                                <div class="r-label">{EXP_LABELS[exp]}</div>
                                <div class="r-pred {cls}" style="font-size: 1.8rem;">{pred}</div>
                                <div class="r-prob" style="font-size: 2.5rem;">{p:.1%}</div>
                                <span class="sev-badge" style="background: {sev_color};">{sev_label}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(res["cam"], caption="Grad-CAM", use_container_width=True)
            else:
                res = run_one(sel_pl, sel_exp, sel_seed)
                if res is None:
                    st.error("Model not available for this combination.")
                elif res.get("skip"):
                    st.warning(f"⚠️ {res['reason']}")
                else:
                    p = res["prob"]
                    sev_label, sev_color = severity_label(p)
                    pred = "POSITIVE" if p >= res["thr"] else "NEGATIVE"
                    cls = "pos" if pred == "POSITIVE" else "neg"
                    border = "result-positive" if pred == "POSITIVE" else "result-negative"
                    p_pct = max(0.0, min(1.0, p)) * 100
                    thr_pct = max(0.0, min(1.0, res["thr"])) * 100
                    cond_note = (" (with 22 hand-crafted features)"
                                 if res.get("is_cond") else "")

                    rcol1, rcol2 = st.columns([1.05, 1])
                    with rcol1:
                        st.markdown(f"""
                        <div class="result-card {border}">
                            <div class="r-label">Diagnosis</div>
                            <div class="r-pred {cls}">{pred}</div>
                            <div class="r-prob-label">Pneumonia Probability</div>
                            <div class="r-prob">{p:.1%}</div>
                            <div><span class="sev-badge" style="background: {sev_color};">
                                Severity · {sev_label}
                            </span></div>
                            <div class="gauge">
                                <div class="gauge-fill" style="width: {p_pct}%;"></div>
                                <div class="gauge-marker" style="left: {thr_pct}%;"></div>
                            </div>
                            <div class="gauge-labels">
                                <span>0%</span>
                                <span style="font-weight: 600; color: #1F2937;">
                                    Threshold: {thr_pct:.1f}%
                                </span>
                                <span>100%</span>
                            </div>
                            <div class="meta-info">
                                <strong>Model:</strong> {sel_pl} · {EXP_LABELS[sel_exp]} · seed {sel_seed}<br>
                                <strong>Architecture:</strong> ResNet-50 (ImageNet pretrained){cond_note}<br>
                                <strong>Input mode:</strong> {res['mode']} · 224×224<br>
                                <strong>Decision:</strong> probability {"≥" if pred == "POSITIVE" else "<"}
                                {res['thr']:.3f} → <strong>{pred}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with rcol2:
                        st.image(res["cam"], caption="🔥 Grad-CAM Attention Map",
                                 use_container_width=True)
                        if res.get("is_cond"):
                            st.markdown("""
                            <div class="info-card">
                                <strong>🧬 Conditional model active.</strong>
                                The 22 hand-crafted radiomic features (mask geometry,
                                ROI intensity stats, GLCM texture) were extracted from
                                this upload using Otsu segmentation and z-score normalized
                                using training-set statistics from the test CSV. Note that
                                feature distributions may differ slightly from training due
                                to using Otsu instead of nnU-Net masks at inference time.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="info-card">
                                <strong>Reading the heatmap:</strong> Brighter regions show where
                                the model focused most strongly during inference. A well-calibrated
                                model should highlight <em>lung parenchyma</em> — not bones, edges,
                                text overlays, or borders. Misplaced attention is a red flag for
                                spurious features.
                            </div>
                            """, unsafe_allow_html=True)


# ============================================================
# RESEARCH METRICS TAB
# ============================================================
with tab_research:
    sub_ov, sub_abl, sub_roc, sub_pr, sub_cm, sub_ext, sub_gc = st.tabs([
        "📋 Overview", "🔬 Ablation", "📈 ROC", "📉 PR",
        "🧮 Confusion", "🌍 External", "🔥 Grad-CAM",
    ])

    # ----- Overview -----
    with sub_ov:
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Total Runs", f"{total_runs}/36")
        with m2: st.metric("Best Mean AUC", f"{best['auc_mean']:.4f}",
                           f"{best['pipeline']} · {EXP_LABELS.get(best['experiment'], best['experiment'])}")
        with m3:
            if "delta" in p_auc.columns:
                st.metric("Max ΔAUC", f"+{p_auc['delta'].max():.4f}")
        with m4: st.metric("Significant (p<0.05)", f"{n_sig}/{len(EXPERIMENTS)}")

        st.markdown("### Per-Run Aggregation (3-seed mean ± std)")
        st.dataframe(display_metrics_table(agg), use_container_width=True, hide_index=True)

        st.markdown("### AUC by Configuration")
        st.pyplot(plot_metric_bars(agg))

        with st.expander("⬇️ Downloads"):
            st.download_button("Per-run results", df_long.to_csv(index=False).encode(),
                               "per_run_results.csv", "text/csv")
            st.download_button("Aggregated", agg.to_csv(index=False).encode(),
                               "aggregated.csv", "text/csv")

    # ----- Ablation -----
    with sub_abl:
        st.markdown("### Ablation: nnU-Net vs Otsu")
        if "delta" not in p_auc.columns:
            st.info("Both pipelines required.")
        else:
            ac1, ac2 = st.columns([2, 1])
            with ac1:
                st.markdown("**Mean AUC ± std with delta**")
                st.dataframe(display_ablation_table(p_str, p_auc),
                             use_container_width=True, hide_index=True)
            with ac2:
                st.markdown("**Paired t-test**")
                st.dataframe(display_ttest_table(ttests),
                             use_container_width=True, hide_index=True)

            st.pyplot(plot_delta_bars(p_auc))
            st.success(
                "**Headline finding:** ΔAUC scales monotonically with the classifier's "
                "structural dependence on the mask. Plain inputs do not benefit from better "
                "segmentation. Masked-ROI inputs benefit substantially "
                "(+0.034 to +0.045 AUC, p<0.05)."
            )

    # ----- ROC -----
    with sub_roc:
        st.markdown("### ROC Curves (3-seed mean)")
        try:
            st.pyplot(plot_roc_overlay(runs, threshold_mode))
        except KeyError as e:
            st.error(f"⚠ Prediction CSV column issue: {e}.")
        except Exception as e:
            st.error(f"ROC plot failed: {e}")

    # ----- PR -----
    with sub_pr:
        st.markdown("### Precision-Recall Curves (3-seed mean)")
        try:
            st.pyplot(plot_pr_overlay(runs))
        except KeyError as e:
            st.error(f"⚠ Prediction CSV column issue: {e}.")
        except Exception as e:
            st.error(f"PR plot failed: {e}")

    # ----- Confusion -----
    with sub_cm:
        st.markdown(f"### Confusion Matrices (threshold = {threshold_mode})")
        try:
            st.pyplot(plot_confusion_grid(runs, threshold_mode))
        except KeyError as e:
            st.error(f"⚠ Prediction CSV column issue: {e}.")
        except Exception as e:
            st.error(f"Confusion plot failed: {e}")

    # ----- External -----
    with sub_ext:
        st.markdown("### External Validation: Kermany Pediatric Pneumonia")
        try:
            ext = pd.read_csv(DEFAULTS["external_summary"])
            per_seed = pd.read_csv(DEFAULTS["external_per_seed"])

            st.markdown("**Summary (zero-shot transfer):**")
            st.dataframe(ext, use_container_width=True)

            rsna_means = agg[agg.experiment == "classifier_plain"].set_index("pipeline")["auc_mean"]
            ext_idx = ext.set_index("pipeline") if "pipeline" in ext.columns else ext
            rows = []
            for pl in ["nnUNet", "Otsu"]:
                if pl in rsna_means.index and pl in ext_idx.index:
                    r = float(rsna_means.loc[pl])
                    k = float(ext_idx.loc[pl, "auc_mean"])
                    rows.append({"Pipeline": pl,
                                 "RSNA AUC": f"{r:.4f}",
                                 "Kermany AUC": f"{k:.4f}",
                                 "Δ": f"{k - r:+.4f}"})
            if rows:
                st.markdown("**Domain shift comparison:**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("**Per-seed results:**")
            st.dataframe(per_seed, use_container_width=True, hide_index=True)
            st.info("AUC went **up** on Kermany — pediatric task is easier and class-balanced. "
                    "Pipeline ranking inverted, suggesting partial dataset specificity.")
        except Exception as e:
            st.warning(f"External CSVs not found: {e}")

    # ----- Grad-CAM (test set) -----
    with sub_gc:
        st.markdown("### Grad-CAM Inspector (Test Set)")
        pl_choice = st.radio("Pipeline", ["nnUNet", "Otsu"], horizontal=True, key="gc_pl")
        csv_path = nn_csv if pl_choice == "nnUNet" else ot_csv

        if not Path(csv_path).exists():
            st.error(f"Test CSV not found: {csv_path}")
        else:
            df_test = load_csv(csv_path)
            cond_features = [c for c in df_test.columns
                             if c.startswith(("roi_", "glcm_", "mask_"))
                             and not c.endswith("_path")]
            cond_dim = len(cond_features)
            label_col = _detect_col(df_test, LABEL_COLS)

            gcc1, gcc2 = st.columns([1, 3])
            with gcc1:
                if label_col:
                    label_filter = st.radio("Filter", ["All", "Positive", "Negative"], key="gc_filt")
                    if label_filter == "Positive":
                        pool = df_test.index[df_test[label_col] == 1].tolist()
                    elif label_filter == "Negative":
                        pool = df_test.index[df_test[label_col] == 0].tolist()
                    else:
                        pool = df_test.index.tolist()
                else:
                    pool = df_test.index.tolist()

                row_idx = st.selectbox("Image index", pool[:300], key="gc_idx")
                same_pl = [(pl, exp, s) for (pl, exp, s) in runs if pl == pl_choice]
                exp_options = sorted(
                    set(exp for _, exp, _ in same_pl),
                    key=lambda e: EXPERIMENTS.index(e) if e in EXPERIMENTS else 999,
                )
                sel_exps = st.multiselect("Configs", exp_options,
                                          default=exp_options[:4], key="gc_cfg")
                sel_seed_gc = st.selectbox("Seed", SEEDS, index=0, key="gc_seed")
                run_btn = st.button("🔥 Generate", type="primary",
                                    use_container_width=True, key="gc_btn")

            with gcc2:
                if run_btn and sel_exps:
                    row = df_test.iloc[row_idx]
                    if label_col:
                        st.markdown(f"**True label:** `{int(row[label_col])}` &nbsp;•&nbsp; "
                                    f"**image_id:** `{row.get('image_id', '?')}`")
                    cols_g = st.columns(min(len(sel_exps), 4))
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    for i, exp in enumerate(sel_exps):
                        key = (pl_choice, exp, sel_seed_gc)
                        if key not in runs:
                            with cols_g[i % len(cols_g)]:
                                st.warning(f"Missing: {exp}")
                            continue
                        is_cond, mode = parse_experiment(exp)
                        this_cd = cond_dim if is_cond else 0
                        try:
                            model = load_model(runs[key]["model_path"], is_cond, this_cd, device)
                        except Exception as e:
                            with cols_g[i % len(cols_g)]:
                                st.error(f"Load: {e}")
                            continue
                        img_t, pil = prepare_input(row, mode, normalize_mode=normalize_mode)
                        img_t = img_t.to(device)
                        cond = None
                        if is_cond and cond_features:
                            cond = torch.tensor(row[cond_features].values.astype(np.float32))\
                                .unsqueeze(0).to(device)
                        cam_eng = GradCAM(model, get_target_layer(model, is_cond))
                        cam = cam_eng(img_t, cond)
                        cam_eng.close()
                        with torch.no_grad():
                            logit = model(img_t, cond) if is_cond else model(img_t)
                            prob = torch.sigmoid(logit).item()
                        overlay = overlay_heatmap(pil, cam)
                        pred = int(prob >= runs[key]["best_thr"])
                        if label_col:
                            badge = "✅" if pred == int(row[label_col]) else "❌"
                        else:
                            badge = ""
                        with cols_g[i % len(cols_g)]:
                            st.markdown(f"**{EXP_LABELS.get(exp, exp)}** {badge}")
                            st.caption(f"prob={prob:.3f} · thr={runs[key]['best_thr']:.3f}")
                            st.image(overlay, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="margin-top: 4rem; padding: 1.5rem; border-top: 1px solid #E5E7EB;
            text-align: center; color: #6B7280; font-size: 0.85rem;">
    <strong>RSNA Pneumonia Detection · Anatomy-Aware Classification</strong><br>
    nnU-Net vs Otsu Ablation Study · 36 trained models · 3-seed validation · RSNA + Kermany external<br>
    <em>Research demo — not for clinical use</em>
</div>
""", unsafe_allow_html=True)