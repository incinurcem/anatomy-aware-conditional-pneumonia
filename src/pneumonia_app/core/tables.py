"""DataFrame display helpers."""
from .config import EXP_LABELS


def display_metrics_table(agg):
    df = agg.copy()
    df["experiment"] = df["experiment"].map(EXP_LABELS).fillna(df["experiment"])
    df = df[["pipeline", "experiment", "n_seeds", "auc_str", "f1_mean", "f1_std"]]
    df.columns = ["Pipeline", "Configuration", "Seeds", "AUC (mean ± std)", "F1 mean", "F1 std"]
    df["F1 mean"] = df["F1 mean"].round(4)
    df["F1 std"]  = df["F1 std"].round(4)
    return df


def display_ablation_table(p_str, p_auc):
    df = p_str.copy()
    if "delta" in p_auc.columns:
        df["Δ AUC"] = p_auc["delta"].apply(lambda x: f"{x:+.4f}")
    df.index = [EXP_LABELS.get(e, e) for e in df.index]
    df.index.name = "Configuration"
    return df.reset_index()


def display_ttest_table(t):
    df = t.copy()
    df["experiment"] = df["experiment"].map(EXP_LABELS).fillna(df["experiment"])
    df["delta"] = df["delta"].apply(lambda x: f"{x:+.4f}")
    df["t"]     = df["t"].apply(lambda x: f"{x:+.3f}")
    df["p"]     = df["p"].apply(lambda x: f"{x:.4f}")
    df = df[["experiment", "delta", "t", "p", "sig"]]
    df.columns = ["Configuration", "ΔAUC", "t-stat", "p-value", "Significance"]
    return df