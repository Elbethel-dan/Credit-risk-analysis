"""
Feature Utilities Module

Includes:
- WoE & IV Calculation for categorical and binned numeric features
- Outlier Removal using IQR method
- Label Encoding
- Scaling (Standardization / Normalization)
- Aggregate Feature Creation per customer
- Date/Time Feature Extraction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ==========================
# 1️⃣ WoE & IV Functions
# ==========================
def calculate_woe(df, feature_col, target_col, event=1):
    """
    Calculate WoE and IV for a single categorical or binned numeric feature.
    """
    grouped = df.groupby(feature_col)[target_col].agg(['count', 'sum']).rename(columns={'sum':'event'})
    grouped['non_event'] = grouped['count'] - grouped['event']

    # Avoid division by zero
    grouped['event'] = grouped['event'].replace(0, 0.5)
    grouped['non_event'] = grouped['non_event'].replace(0, 0.5)

    # Proportions
    event_total = grouped['event'].sum()
    non_event_total = grouped['non_event'].sum()
    grouped['event_rate'] = grouped['event'] / event_total
    grouped['non_event_rate'] = grouped['non_event'] / non_event_total

    # WoE
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])

    # IV
    grouped['iv_contribution'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
    iv = grouped['iv_contribution'].sum()

    woe_df = grouped.reset_index()[[feature_col, 'count', 'event', 'non_event', 'woe', 'iv_contribution']]
    return woe_df, iv


def woe_summary(df, target_col, numerical_cols=None, categorical_cols=None, bins=5):
    """
    Compute WoE and IV for multiple features (categorical + numeric).
    Numeric columns are automatically binned before WoE calculation.
    """
    df_ = df.copy()
    woe_details = {}
    iv_summary = []

    # Bin numerical columns
    if numerical_cols:
        for col in numerical_cols:
            try:
                df_[col+'_bin'] = pd.qcut(df_[col], q=bins, duplicates='drop')
            except Exception as e:
                print(f"⚠️ Could not bin column {col}: {e}")
        numerical_cols = [col+'_bin' for col in numerical_cols]

    # Combine features
    features = (categorical_cols or []) + (numerical_cols or [])

    # Calculate WoE & IV
    for feature in features:
        woe_df, iv = calculate_woe(df_, feature, target_col)
        woe_details[feature] = woe_df
        iv_summary.append({'feature': feature, 'iv': iv})

    iv_summary_df = pd.DataFrame(iv_summary).sort_values(by='iv', ascending=False).reset_index(drop=True)
    return iv_summary_df, woe_details

# ==========================
# 2️⃣ Outlier Handling
# ==========================
def remove_outliers_iqr(df, numeric_cols=None, factor=1.5, verbose=True):
    """
    Remove outliers from numeric columns using the IQR method.
    """
    df_clean = df.copy()
    if numeric_cols is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    mask = pd.Series(True, index=df_clean.index)
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask &= df_clean[col].between(lower, upper)

    removed_count = len(df_clean) - mask.sum()
    if verbose:
        print(f"⚠️ Removed {removed_count} outlier rows ({removed_count/len(df_clean)*100:.2f}%)")

    return df_clean[mask].copy()

# ==========================
# 3️⃣ Label Encoding
# ==========================
def label_encode_columns(df, categorical_cols=None):
    """
    Label encode categorical columns.
    """
    df_encoded = df.copy()
    encoders = {}

    if categorical_cols is None:
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    return df_encoded, encoders

# ==========================
# 4️⃣ Scaling (Standardization / Normalization)
# ==========================
def scale_columns(df, numeric_cols=None, method='standard'):
    """
    Scale numeric columns using Standardization or Normalization.
    """
    df_scaled = df.copy()
    scalers = {}

    if numeric_cols is None:
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")

        df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
        scalers[col] = scaler

    return df_scaled, scalers

# ==========================
# 5️⃣ Feature Engineering
# ==========================
def aggregate_features(df, customer_id_col, amount_col, numeric_cols=None, categorical_cols=None):
    """
    Aggregate transaction features per customer.
    """
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    agg_dict = {amount_col: ['sum', 'mean', 'count', 'std']}
    for col in numeric_cols:
        if col != amount_col:
            agg_dict[col] = ['mean', 'std', 'sum', 'min', 'max']
    for col in categorical_cols:
        agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else np.nan

    df_agg = df.groupby(customer_id_col).agg(agg_dict)
    df_agg.columns = ['_'.join([str(c) for c in col]).strip('_') for col in df_agg.columns.values]
    return df_agg.reset_index()

def extract_date_features(df, date_col):
    """
    Extract date/time features from a datetime column.
    """
    df_ = df.copy()
    df_[date_col] = pd.to_datetime(df_[date_col], errors='coerce')

    df_['transaction_hour'] = df_[date_col].dt.hour
    df_['transaction_day'] = df_[date_col].dt.day
    df_['transaction_month'] = df_[date_col].dt.month
    df_['transaction_year'] = df_[date_col].dt.year
    df_['transaction_dayofweek'] = df_[date_col].dt.dayofweek
    df_['transaction_quarter'] = df_[date_col].dt.quarter
    df_['is_weekend'] = df_['transaction_dayofweek'].isin([5,6]).astype(int)
    df_['is_business_hours'] = ((df_['transaction_hour'] >= 9) & (df_['transaction_hour'] <= 17)).astype(int)

    return df_.drop(columns=[date_col])
