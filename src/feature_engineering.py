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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


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
        X["transaction_dayofweek"] = X[self.date_col].dt.dayofweek
        X["is_weekend"] = X["transaction_dayofweek"].isin([5, 6]).astype(int)

        X.drop(columns=[self.date_col], inplace=True)

        self.feature_names_ = X.columns.tolist()
        return X


# ==========================================================
# 2. CUSTOMER AGGREGATION
# ==========================================================

class AggregateFeatures(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        customer_id_col: str,
        amount_col: str,
        numeric_cols: Optional[List[str]] = None,
    ):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.numeric_cols = numeric_cols or []
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        agg_dict = {
            self.amount_col: ["sum", "mean", "count", "std"]
        }

        for col in self.numeric_cols:
            if col != self.amount_col:
                agg_dict[col] = ["mean", "std"]

        result = X.groupby(self.customer_id_col).agg(agg_dict)

        result.columns = [
            "_".join(col).strip("_") for col in result.columns.values
        ]

        result.reset_index(inplace=True)

        self.feature_names_ = result.columns.tolist()
        return result


# ==========================================================
# 3. FEATURE INTERACTIONS
# ==========================================================

class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Coefficient of variation
        for col in X.columns:
            if col.endswith("_std"):
                mean_col = col.replace("_std", "_mean")
                if mean_col in X.columns:
                    X[col.replace("_std", "_cv")] = (
                        X[col] / (X[mean_col] + 1e-10)
                    )

        X.fillna(0, inplace=True)
        return X


# ==========================================================
# 4. SAFE OUTLIER CLIPPING (No Row Removal)
# ==========================================================

class ClipOutliers(BaseEstimator, TransformerMixin):

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (
                Q1 - self.factor * IQR,
                Q3 + self.factor * IQR,
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower, upper)
        return X


# ==========================================================
# 5. PREPROCESSOR
# ==========================================================

class DynamicPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, scaling_method: str = "standard"):
        self.scaling_method = scaling_method
        self.preprocessor_ = None

    def fit(self, X: pd.DataFrame, y=None):

        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(exclude=np.number).columns

        scaler = (
            StandardScaler()
            if self.scaling_method == "standard"
            else MinMaxScaler()
        )

        self.preprocessor_ = ColumnTransformer(
            transformers=[
                ("num", scaler, numeric_features),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="first",
                    ),
                    categorical_features,
                ),
            ]
        )

        self.preprocessor_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.preprocessor_.transform(X)
        feature_names = self.preprocessor_.get_feature_names_out()

        return pd.DataFrame(transformed, columns=feature_names, index=X.index)


# ==========================================================
# 6. PIPELINE BUILDER 
# ==========================================================

def build_feature_engineering_pipeline(
    customer_id_col: str,
    amount_col: str,
    date_col: str,
    numeric_cols: Optional[List[str]] = None,
    scaling_method: str = "standard",
):

    return Pipeline(
        steps=[
            ("date_features", ExtractDateFeatures(date_col)),
            ("aggregation", AggregateFeatures(customer_id_col, amount_col, numeric_cols)),
            ("feature_engineering", FeatureEngineering()),
            ("clip_outliers", ClipOutliers()),
            ("preprocessing", DynamicPreprocessor(scaling_method)),
        ]
    )
