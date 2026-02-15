"""
WoE & Information Value Transformer
Separated to prevent leakage and allow controlled training.
"""

import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from xverse.transformer import WOE


class WoEFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        iv_threshold: float = 0.02,
        max_bins: int = 5,
        min_bin_pct: float = 0.05,
    ):
        self.iv_threshold = iv_threshold
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.woe_ = None
        self.iv_summary_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):

        self.woe_ = WOE(
            iv_threshold=self.iv_threshold,
            max_bins=self.max_bins,
            min_bin_pct=self.min_bin_pct,
        )

        self.woe_.fit(X, y)

        if hasattr(self.woe_, "iv_df_"):
            self.iv_summary_ = self.woe_.iv_df_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.woe_.transform(X)

    def get_iv_summary(self) -> Optional[pd.DataFrame]:
        return self.iv_summary_
