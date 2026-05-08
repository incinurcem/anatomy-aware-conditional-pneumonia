"""
export_final_tables.py
#d
Purpose
-------
Aggregate all experiment outputs and export final summary tables
for report, paper, and deployment evaluation.

Inputs
------
outputs/
    seg_metrics_*.csv
    classifier_metrics.csv
    radiomics_features.csv
    burden_scores.csv
    calibration_metrics.csv
    uncertainty_metrics.csv

Outputs
-------
outputs/final_tables/

    segmentation_summary.csv
    classifier_summary.csv
    radiomics_summary.csv
    burden_summary.csv
    calibration_summary.csv
    uncertainty_summary.csv

    segmentation_table_latex.txt
    classifier_table_latex.txt
"""

import os
import pandas as pd


# ----------------------------------------------------------
# helpers
# ----------------------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------------------------------------------------------
# segmentation summary
# ----------------------------------------------------------

def build_segmentation_summary(seg_dir):

    files = [
        f for f in os.listdir(seg_dir)
        if f.startswith("seg_metrics")
    ]

    dfs = []

    for f in files:

        path = os.path.join(seg_dir, f)

        df = pd.read_csv(path)

        model = f.replace("seg_metrics_", "").replace(".csv", "")

        df["model"] = model

        dfs.append(df)

    final = pd.concat(dfs)

    summary = final.groupby("model").mean().reset_index()

    return summary


# ----------------------------------------------------------
# classifier summary
# ----------------------------------------------------------

def build_classifier_summary(path):

    df = pd.read_csv(path)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]

    summary = df[metrics].mean().to_frame().T

    return summary


# ----------------------------------------------------------
# radiomics summary
# ----------------------------------------------------------

def build_radiomics_summary(path):

    df = pd.read_csv(path)

    summary = df.describe().T

    return summary


# ----------------------------------------------------------
# burden summary
# ----------------------------------------------------------

def build_burden_summary(path):

    df = pd.read_csv(path)

    summary = df.describe()

    return summary


# ----------------------------------------------------------
# calibration summary
# ----------------------------------------------------------

def build_calibration_summary(path):

    df = pd.read_csv(path)

    summary = df.describe()

    return summary


# ----------------------------------------------------------
# uncertainty summary
# ----------------------------------------------------------

def build_uncertainty_summary(path):

    df = pd.read_csv(path)

    summary = df.describe()

    return summary


# ----------------------------------------------------------
# latex export
# ----------------------------------------------------------

def save_latex(df, path):

    latex = df.to_latex(
        index=True,
        float_format="%.4f"
    )

    with open(path, "w") as f:
        f.write(latex)


# ----------------------------------------------------------
# main
# ----------------------------------------------------------

def main():

    base_dir = "outputs"

    final_dir = os.path.join(base_dir, "final_tables")

    ensure_dir(final_dir)

    # -----------------------------
    # segmentation
    # -----------------------------

    seg_summary = build_segmentation_summary(base_dir)

    seg_csv = os.path.join(final_dir, "segmentation_summary.csv")

    seg_summary.to_csv(seg_csv, index=False)

    save_latex(
        seg_summary,
        os.path.join(final_dir, "segmentation_table_latex.txt")
    )

    print("Segmentation summary saved.")

    # -----------------------------
    # classifier
    # -----------------------------

    clf_path = os.path.join(base_dir, "classifier_metrics.csv")

    if os.path.exists(clf_path):

        clf_summary = build_classifier_summary(clf_path)

        clf_csv = os.path.join(final_dir, "classifier_summary.csv")

        clf_summary.to_csv(clf_csv, index=False)

        save_latex(
            clf_summary,
            os.path.join(final_dir, "classifier_table_latex.txt")
        )

        print("Classifier summary saved.")

    # -----------------------------
    # radiomics
    # -----------------------------

    radiomics_path = os.path.join(base_dir, "radiomics_features.csv")

    if os.path.exists(radiomics_path):

        radiomics_summary = build_radiomics_summary(radiomics_path)

        radiomics_summary.to_csv(
            os.path.join(final_dir, "radiomics_summary.csv")
        )

        print("Radiomics summary saved.")

    # -----------------------------
    # burden
    # -----------------------------

    burden_path = os.path.join(base_dir, "burden_scores.csv")

    if os.path.exists(burden_path):

        burden_summary = build_burden_summary(burden_path)

        burden_summary.to_csv(
            os.path.join(final_dir, "burden_summary.csv")
        )

        print("Burden summary saved.")

    # -----------------------------
    # calibration
    # -----------------------------

    cal_path = os.path.join(base_dir, "calibration_metrics.csv")

    if os.path.exists(cal_path):

        cal_summary = build_calibration_summary(cal_path)

        cal_summary.to_csv(
            os.path.join(final_dir, "calibration_summary.csv")
        )

        print("Calibration summary saved.")

    # -----------------------------
    # uncertainty
    # -----------------------------

    unc_path = os.path.join(base_dir, "uncertainty_metrics.csv")

    if os.path.exists(unc_path):

        unc_summary = build_uncertainty_summary(unc_path)

        unc_summary.to_csv(
            os.path.join(final_dir, "uncertainty_summary.csv")
        )

        print("Uncertainty summary saved.")


if __name__ == "__main__":
    main()