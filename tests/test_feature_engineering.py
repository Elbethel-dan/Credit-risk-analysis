# test_feature_engineering.py

import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import (
    calculate_woe, woe_summary,
    remove_outliers_iqr,
    label_encode_columns,
    scale_columns,
    aggregate_features,
    extract_date_features
)

# -----------------------------
# Sample Data
# -----------------------------
data = pd.DataFrame({
    'CustomerId': ['C1','C1','C2','C2','C3'],
    'Amount': [100, 200, 300, 400, 500],
    'Value': [10, 20, 30, 40, 50],
    'ProductCategory': ['A','A','B','B','C'],
    'TransactionTime': pd.to_datetime([
        '2023-01-01 10:00','2023-01-01 15:00',
        '2023-01-02 09:00','2023-01-02 18:00',
        '2023-01-03 12:00'
    ]),
    'FraudResult': [0,1,0,1,0]
})

# -----------------------------
# Test WoE & IV
# -----------------------------
def test_calculate_woe():
    woe_df, iv = calculate_woe(data, 'ProductCategory', 'FraudResult')
    assert isinstance(woe_df, pd.DataFrame)
    assert iv >= 0
    assert 'woe' in woe_df.columns
    assert 'iv_contribution' in woe_df.columns

def test_woe_summary():
    iv_summary_df, woe_details = woe_summary(data, 'FraudResult', numerical_cols=['Amount'], categorical_cols=['ProductCategory'], bins=2)
    assert isinstance(iv_summary_df, pd.DataFrame)
    assert isinstance(woe_details, dict)
    assert 'Amount_bin' in woe_details

# -----------------------------
# Test Outlier Removal
# -----------------------------
def test_remove_outliers_iqr():
    df_clean = remove_outliers_iqr(data, numeric_cols=['Amount'], factor=1.0, verbose=False)
    assert df_clean['Amount'].max() <= data['Amount'].max()
    assert df_clean['Amount'].min() >= data['Amount'].min()

# -----------------------------
# Test Label Encoding
# -----------------------------
def test_label_encode_columns():
    df_encoded, encoders = label_encode_columns(data, categorical_cols=['ProductCategory'])
    assert isinstance(df_encoded, pd.DataFrame)
    assert set(df_encoded['ProductCategory'].unique()) <= set(range(len(data['ProductCategory'].unique())))
    assert isinstance(encoders['ProductCategory'], type(encoders['ProductCategory']))

# -----------------------------
# Test Scaling
# -----------------------------
def test_scale_columns():
    df_scaled, scalers = scale_columns(data, numeric_cols=['Amount','Value'], method='standard')
    assert abs(df_scaled['Amount'].mean()) < 1e-6 or np.isnan(df_scaled['Amount'].mean()) == False
    assert abs(df_scaled['Value'].mean()) < 1e-6 or np.isnan(df_scaled['Value'].mean()) == False
    df_minmax, _ = scale_columns(data, numeric_cols=['Amount','Value'], method='minmax')
    assert df_minmax['Amount'].max() <= 1
    assert df_minmax['Value'].min() >= 0

# -----------------------------
# Test Aggregate Features
# -----------------------------
def test_aggregate_features():
    df_agg = aggregate_features(data, 'CustomerId', 'Amount', numeric_cols=['Value'], categorical_cols=['ProductCategory'])
    assert 'Amount_sum' in df_agg.columns
    assert 'Value_mean' in df_agg.columns
    assert df_agg['CustomerId'].nunique() == 3

# -----------------------------
# Test Date Features
# -----------------------------
def test_extract_date_features():
    df_features = extract_date_features(data, 'TransactionTime')
    expected_cols = ['transaction_hour','transaction_day','transaction_month','transaction_year',
                     'transaction_dayofweek','transaction_quarter','is_weekend','is_business_hours']
    for col in expected_cols:
        assert col in df_features.columns
    assert df_features.shape[0] == data.shape[0]
