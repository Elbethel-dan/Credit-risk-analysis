"""
Core Feature Engineering Pipeline
Includes:
- Date feature extraction
- Customer aggregation
- Feature interactions
- Outlier clipping 
- Encoding and scaling
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from typing import Optional
from xverse.transformer import WOE


# ==========================================================
# 1. DATE FEATURE EXTRACTION
# ==========================================================

class ExtractDateFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, date_col: str):
        self.date_col = date_col
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")

        X["transaction_hour"] = X[self.date_col].dt.hour
        X["transaction_day"] = X[self.date_col].dt.day
        X["transaction_month"] = X[self.date_col].dt.month
        X["transaction_year"] = X[self.date_col].dt.year
        X["transaction_dayofweek"] = X[self.date_col].dt.dayofweek
        X["is_weekend"] = X["transaction_dayofweek"].isin([5, 6]).astype(int)

        X.drop(columns=[self.date_col], inplace=True)

        self.feature_names_ = X.columns.tolist()
        return X


# ==========================================================
# 2. CUSTOMER AGGREGATION
# ==========================================================
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col: str, amount_col: str, target_col: str = None):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        agg = (
            X.groupby(self.customer_id_col)[self.amount_col]
            .agg(["sum", "mean", "count", "std"])
            .reset_index()
        )

        agg.columns = [
            self.customer_id_col,
            "total_transaction_amount",
            "average_transaction_amount",
            "transaction_count",
            "std_transaction_amount",
        ]

        # Merge back to original data
        X = X.merge(agg, on=self.customer_id_col, how="left")

        return X


# ==========================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ==========================================================
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes categorical columns using Label Encoding to reduce dimensionality.
    """

    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.encoders = {}  # dictionary to store LabelEncoders for each column

    def fit(self, X, y=None):
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))  # ensure all data is string
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders.items():
            X[col] = le.transform(X[col].astype(str))
        return X

# ==========================================================
# 4. WOE TRANSFORMER
# ==========================================================
class WoEFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.woe_ = None
        self.iv_summary_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.woe_ = WOE()  # no arguments
        self.woe_.fit(X, y)

        if hasattr(self.woe_, "iv_df_"):
            self.iv_summary_ = self.woe_.iv_df_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.woe_.transform(X)

    def get_iv_summary(self) -> Optional[pd.DataFrame]:
        return self.iv_summary_


# ==========================================================
# 5. OUTLIER REMOVAL
# ==========================================================
class RemoveOutliers(BaseEstimator, TransformerMixin):

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include="number").columns

        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR
            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X = X[(X[col] >= lower) & (X[col] <= upper)]

        return X.reset_index(drop=True)


# ==========================================================
# 6. SCALING TRANSFORMER
# ==========================================================
class ScaleNumericalFeatures(BaseEstimator, TransformerMixin):
    """
    Scale numerical features using either Standardization or Normalization.

    Parameters
    ----------
    method : str, default="standard"
        "standard" -> StandardScaler (mean=0, std=1)
        "normalize" -> MinMaxScaler (range [0,1])
    columns : list of str, optional
        List of columns to scale. If None, all numeric columns are scaled.
    """
    def __init__(self, method: str = "standard", columns: Optional[list] = None):
        self.method = method
        self.columns = columns
        self.scaler = None
        self.feature_names_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        self.feature_names_ = self.columns or numeric_cols

        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "normalize":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'normalize'")

        self.scaler.fit(X[self.feature_names_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.feature_names_] = self.scaler.transform(X[self.feature_names_])
        return X


# ==========================================================
# 7. PIPELINE BUILDER 
# ==========================================================

def build_feature_engineering_pipeline(
    customer_id_col: str,
    amount_col: str,
    date_col: str,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    scaling_method: str = "standard",
    iv_threshold: float = 0.02,
    max_bins: int = 5,
    min_bin_pct: float = 0.05,
):
    """
    Feature Engineering Pipeline (your specified order):
    1. Date feature extraction
    2. Categorical encoding
    3. WoE transformation
    4. Outlier removal
    5. Scaling
    """
    return Pipeline(
        steps=[
            ("date_features", ExtractDateFeatures(date_col)),
            ("encode", LabelEncoderTransformer(categorical_cols or [])),
            ("woe", WoEFeatureTransformer()),
            ("remove_outliers", RemoveOutliers()),
            ("scaling", ScaleNumericalFeatures(method=scaling_method)),
        ]
    )
