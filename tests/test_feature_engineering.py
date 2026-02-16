import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import build_feature_engineering_pipeline


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "CustomerId": [1, 1, 2, 2, 3],
        "Amount": [100, 200, 150, 300, 400],
        "Value": [10, 20, 15, 30, 40],
        "TransactionStartTime": pd.date_range(
            start="2023-01-01",
            periods=5,
            freq="D"
        ),
        "Category": ["A", "B", "A", "B", "C"]
    })


def test_pipeline_runs(sample_df):

    pipeline = build_feature_engineering_pipeline(
        customer_id_col="CustomerId",
        amount_col="Amount",
        date_col="TransactionStartTime",
        numeric_cols=["Value"],
        scaling_method="standard",
    )

    X_transformed = pipeline.fit_transform(sample_df)

    # Should return DataFrame
    assert isinstance(X_transformed, pd.DataFrame)

    # Should not be empty
    assert X_transformed.shape[0] > 0
    assert X_transformed.shape[1] > 0


def test_no_row_loss_after_clipping(sample_df):

    pipeline = build_feature_engineering_pipeline(
        customer_id_col="CustomerId",
        amount_col="Amount",
        date_col="TransactionStartTime",
        numeric_cols=["Value"],
    )

    X_transformed = pipeline.fit_transform(sample_df)

    # Because of aggregation, rows = unique customers
    assert X_transformed.shape[0] == sample_df["CustomerId"].nunique()


def test_pipeline_transform_after_fit(sample_df):

    pipeline = build_feature_engineering_pipeline(
        customer_id_col="CustomerId",
        amount_col="Amount",
        date_col="TransactionStartTime",
        numeric_cols=["Value"],
    )

    pipeline.fit(sample_df)
    X_transformed = pipeline.transform(sample_df)

    assert isinstance(X_transformed, pd.DataFrame)
