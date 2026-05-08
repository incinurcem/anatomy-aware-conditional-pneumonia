"""Statistical helpers."""
import numpy as np
import pandas as pd
from scipy import stats as sst
from sklearn.metrics import roc_auc_score


def paired_ttest(df, exp):
    nn = df[(df.experiment == exp) & (df.pipeline == "nnUNet")].sort_values("seed")
    ot = df[(df.experiment == exp) & (df.pipeline == "Otsu")].sort_values("seed")
    if len(nn) < 2 or len(nn) != len(ot):
        return None
    t, p = sst.ttest_rel(nn["auc_youden"].values, ot["auc_youden"].values)
    delta = nn["auc_youden"].mean() - ot["auc_youden"].mean()
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return {"t": float(t), "p": float(p), "delta": float(delta), "n": len(nn), "sig": sig}


def all_ttests(df):
    rows = []
    for exp in df["experiment"].unique():
        r = paired_ttest(df, exp)
        if r:
            rows.append({"experiment": exp, **r})
    return pd.DataFrame(rows)


def bootstrap_auc_ci(y_true, y_prob, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    a = np.array(aucs)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))