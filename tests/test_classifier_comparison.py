import pytest
import pandas as pd
import num

#d

REQUIRED_COLUMNS = [
    "model_name",
    "accuracy",
    "auc",
    "precision",
    "recall",
    "f1",
    "specificity"
]


def validate_classifier_comparison_df(df: pd.DataFrame) -> None:
    """
    Validates the structure of a classifier comparison DataFrame.
    Raises ValueError if the DataFrame is invalid.
    """
    if df is None:
        raise ValueError("Comparison DataFrame is None.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Comparison DataFrame is empty.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["model_name"].isnull().any():
        raise ValueError("model_name column contains null values.")

    metric_cols = [c for c in REQUIRED_COLUMNS if c != "model_name"]
    for col in metric_cols:
        if df[col].isnull().any():
            raise ValueError(f"{col} contains null values.")

        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"{col} must be numeric.")

        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"{col} must be between 0 and 1.")


def sort_models_by_metric(df: pd.DataFrame, metric: str = "auc", ascending: bool = False) -> pd.DataFrame:
    """
    Sorts classifier comparison DataFrame by a given metric.
    """
    validate_classifier_comparison_df(df)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame.")

    return df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)


def get_best_model(df: pd.DataFrame, priority_metrics=None) -> pd.Series:
    """
    Returns the best model row according to priority metrics.
    Default priority:
    1. auc
    2. f1
    3. recall
    """
    validate_classifier_comparison_df(df)

    if priority_metrics is None:
        priority_metrics = ["auc", "f1", "recall"]

    for metric in priority_metrics:
        if metric not in df.columns:
            raise ValueError(f"Priority metric '{metric}' not found in DataFrame.")

    sorted_df = df.sort_values(
        by=priority_metrics,
        ascending=[False] * len(priority_metrics)
    ).reset_index(drop=True)

    return sorted_df.iloc[0]


@pytest.fixture
def sample_comparison_df():
    return pd.DataFrame({
        "model_name": ["resnet18", "efficientnet_b0", "densenet121", "convnext_tiny"],
        "accuracy": [0.82, 0.85, 0.81, 0.84],
        "auc": [0.88, 0.91, 0.87, 0.90],
        "precision": [0.80, 0.84, 0.79, 0.83],
        "recall": [0.78, 0.86, 0.77, 0.82],
        "f1": [0.79, 0.85, 0.78, 0.825],
        "specificity": [0.84, 0.85, 0.83, 0.86]
    })


def test_validate_classifier_comparison_df_valid(sample_comparison_df):
    """
    Valid comparison dataframe should pass without exception.
    """
    validate_classifier_comparison_df(sample_comparison_df)


def test_validate_classifier_comparison_df_missing_column(sample_comparison_df):
    """
    Missing required columns should raise ValueError.
    """
    bad_df = sample_comparison_df.drop(columns=["auc"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_classifier_comparison_df(bad_df)


def test_validate_classifier_comparison_df_empty():
    """
    Empty dataframe should raise ValueError.
    """
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        validate_classifier_comparison_df(empty_df)


def test_validate_classifier_comparison_df_null_model_name(sample_comparison_df):
    """
    Null model names should raise ValueError.
    """
    bad_df = sample_comparison_df.copy()
    bad_df.loc[1, "model_name"] = None

    with pytest.raises(ValueError, match="model_name"):
        validate_classifier_comparison_df(bad_df)


def test_validate_classifier_comparison_df_metric_out_of_range(sample_comparison_df):
    """
    Metric values outside [0, 1] should raise ValueError.
    """
    bad_df = sample_comparison_df.copy()
    bad_df.loc[0, "auc"] = 1.2

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_classifier_comparison_df(bad_df)


def test_validate_classifier_comparison_df_metric_non_numeric(sample_comparison_df):
    """
    Non-numeric metric values should raise ValueError.
    """
    bad_df = sample_comparison_df.copy()
    bad_df["auc"] = ["x", "y", "z", "w"]

    with pytest.raises(ValueError, match="must be numeric"):
        validate_classifier_comparison_df(bad_df)


def test_sort_models_by_auc(sample_comparison_df):
    """
    Models should be sorted descending by AUC by default.
    """
    sorted_df = sort_models_by_metric(sample_comparison_df, metric="auc")

    assert sorted_df.iloc[0]["model_name"] == "efficientnet_b0"
    assert sorted_df.iloc[1]["model_name"] == "convnext_tiny"
    assert sorted_df.iloc[-1]["model_name"] == "densenet121"


def test_sort_models_by_accuracy(sample_comparison_df):
    """
    Models should be sorted descending by accuracy.
    """
    sorted_df = sort_models_by_metric(sample_comparison_df, metric="accuracy")

    assert sorted_df.iloc[0]["model_name"] == "efficientnet_b0"
    assert sorted_df.iloc[-1]["model_name"] == "densenet121"


def test_sort_models_by_metric_invalid_metric(sample_comparison_df):
    """
    Invalid metric name should raise ValueError.
    """
    with pytest.raises(ValueError, match="not found"):
        sort_models_by_metric(sample_comparison_df, metric="mae")


def test_get_best_model_default_priority(sample_comparison_df):
    """
    Best model should be selected by auc > f1 > recall.
    """
    best_row = get_best_model(sample_comparison_df)

    assert best_row["model_name"] == "efficientnet_b0"
    assert best_row["auc"] == pytest.approx(0.91)
    assert best_row["f1"] == pytest.approx(0.85)


def test_get_best_model_custom_priority(sample_comparison_df):
    """
    Best model should follow custom priority metrics.
    """
    custom_df = sample_comparison_df.copy()
    custom_df.loc[3, "recall"] = 0.90

    best_row = get_best_model(custom_df, priority_metrics=["recall", "auc"])

    assert best_row["model_name"] == "convnext_tiny"
    assert best_row["recall"] == pytest.approx(0.90)


def test_get_best_model_invalid_priority_metric(sample_comparison_df):
    """
    Invalid custom priority metric should raise ValueError.
    """
    with pytest.raises(ValueError, match="Priority metric"):
        get_best_model(sample_comparison_df, priority_metrics=["ece"])


def test_sort_models_by_metric_ascending(sample_comparison_df):
    """
    Ascending sort should return worst model first.
    """
    sorted_df = sort_models_by_metric(sample_comparison_df, metric="auc", ascending=True)

    assert sorted_df.iloc[0]["model_name"] == "densenet121"
    assert sorted_df.iloc[-1]["model_name"] == "efficientnet_b0"


def test_get_best_model_tie_breaking():
    """
    Tie should be resolved using next priority metrics.
    """
    df = pd.DataFrame({
        "model_name": ["model_a", "model_b", "model_c"],
        "accuracy": [0.80, 0.80, 0.80],
        "auc": [0.90, 0.90, 0.88],
        "precision": [0.81, 0.82, 0.79],
        "recall": [0.85, 0.83, 0.82],
        "f1": [0.84, 0.86, 0.80],
        "specificity": [0.79, 0.80, 0.81]
    })

    best_row = get_best_model(df, priority_metrics=["auc", "f1", "recall"])

    assert best_row["model_name"] == "model_b"


def test_validate_classifier_comparison_df_numpy_float_types():
    """
    Numpy float dtypes should be accepted.
    """
    df = pd.DataFrame({
        "model_name": ["m1", "m2"],
        "accuracy": np.array([0.8, 0.9], dtype=np.float32),
        "auc": np.array([0.82, 0.93], dtype=np.float64),
        "precision": np.array([0.78, 0.88], dtype=np.float32),
        "recall": np.array([0.79, 0.87], dtype=np.float64),
        "f1": np.array([0.785, 0.875], dtype=np.float32),
        "specificity": np.array([0.81, 0.90], dtype=np.float64),
    })

    validate_classifier_comparison_df(df)


def test_validate_classifier_comparison_df_duplicate_model_names_allowed():
    """
    Duplicate model names are allowed if comparison includes repeated runs.
    """
    df = pd.DataFrame({
        "model_name": ["resnet18", "resnet18", "efficientnet_b0"],
        "accuracy": [0.80, 0.82, 0.85],
        "auc": [0.86, 0.88, 0.91],
        "precision": [0.79, 0.81, 0.84],
        "recall": [0.77, 0.79, 0.86],
        "f1": [0.78, 0.80, 0.85],
        "specificity": [0.82, 0.83, 0.85]
    })

    validate_classifier_comparison_df(df)


def test_get_best_model_single_row():
    """
    A single-row dataframe should return that row.
    """
    df = pd.DataFrame({
        "model_name": ["only_model"],
        "accuracy": [0.91],
        "auc": [0.95],
        "precision": [0.90],
        "recall": [0.92],
        "f1": [0.91],
        "specificity": [0.89]
    })

    best_row = get_best_model(df)

    assert best_row["model_name"] == "only_model"
    assert best_row["auc"] == pytest.approx(0.95)