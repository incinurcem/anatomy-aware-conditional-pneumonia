"""Data loading, caching, and aggregation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .config import EXPERIMENTS, SEEDS


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def gather_runs(nn_root, ot_root):
    """Walk both pipelines × experiments × seeds; return runs dict + long DataFrame."""
    roots = {"nnUNet": nn_root, "Otsu": ot_root}
    runs, rows = {}, []
    for pl, root in roots.items():
        for exp in EXPERIMENTS:
            for seed in SEEDS:
                d = Path(root) / exp / f"seed_{seed}"
                mp, pp, wp = d / "test_metrics.json", d / "test_predictions.csv", d / "best_model.pth"
                if not mp.exists():
                    continue
                m = json.loads(mp.read_text())
                m_y, m_5 = m["test_metrics_threshold_youden"], m["test_metrics_threshold_0.5"]
                runs[(pl, exp, seed)] = {
                    "metrics": m, "y": m_y, "p5": m_5,
                    "preds": pd.read_csv(pp) if pp.exists() else None,
                    "model_path": str(wp),
                    "best_thr": m.get("youden_threshold_from_val", 0.5),
                }
                rows.append({
                    "pipeline": pl, "experiment": exp, "seed": seed,
                    "auc_youden": m_y["roc_auc"],
                    "f1_youden":  m_y["f1"],
                    "auc_05":     m_5["roc_auc"],
                    "f1_05":      m_5["f1"],
                    "thr_youden": m_y["threshold"],
                })
    return runs, pd.DataFrame(rows)


def aggregate_seed(df):
    g = df.groupby(["pipeline", "experiment"])
    agg = g.agg(
        n_seeds=("seed", "count"),
        auc_mean=("auc_youden", "mean"),
        auc_std =("auc_youden", "std"),
        f1_mean =("f1_youden", "mean"),
        f1_std  =("f1_youden", "std"),
    ).reset_index()
    agg["auc_str"] = agg.apply(
        lambda r: f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}" if r["n_seeds"] > 1
                  else f"{r['auc_mean']:.4f}", axis=1)
    return agg


def ablation_pivot(agg):
    p_auc = agg.pivot(index="experiment", columns="pipeline", values="auc_mean")
    p_str = agg.pivot(index="experiment", columns="pipeline", values="auc_str")
    if "nnUNet" in p_auc.columns and "Otsu" in p_auc.columns:
        p_auc["delta"] = p_auc["nnUNet"] - p_auc["Otsu"]
    order = {e: i for i, e in enumerate(EXPERIMENTS)}
    p_auc = p_auc.reindex(sorted(p_auc.index, key=lambda e: order.get(e, 999)))
    p_str = p_str.reindex(sorted(p_str.index, key=lambda e: order.get(e, 999)))
    return p_auc, p_str