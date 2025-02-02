"""
Data preprocessing pipeline: handles cleaning, missing values, outliers,
and calls FeatureEngineer for advanced features.
No database creation or usage.
"""

import os
import pandas as pd
import numpy as np
# 1. Enable the experimental feature
from sklearn.experimental import enable_iterative_imputer
# 2. Now safely import the imputer classes
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler

from .feature_engineering import FeatureEngineer


class DataPreprocessor:
    """
    DataPreprocessor cleans raw data (missing values, outliers) 
    and calls FeatureEngineer for feature creation.
    """

    def __init__(self):
        """
        No database initialization here, just set up feature engineering instance.
        """
        self.feature_engineer = FeatureEngineer()

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates data cleaning steps:
        1) Handle missing values
        2) Remove outliers
        """
        data = self._handle_missing_values(data)
        data = self._remove_outliers(data)
        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with:
         - Synthetic fill for completely missing columns
         - IterativeImputer for high-missing columns
         - KNNImputer for low-missing columns
         - Final forward/backward fill
        """
        data = data.copy()
        missing_pct = data.isnull().sum() / len(data)

        # 1) If any columns are 100% missing, fill with synthetic data
        fully_missing = missing_pct[missing_pct == 1.0].index
        for col in fully_missing:
            data[col] = np.random.normal(0, 1, size=len(data))

        # 2) Group columns by missing percentage
        high_missing = missing_pct[(missing_pct > 0.3) & (missing_pct < 1.0)].index
        low_missing  = missing_pct[(missing_pct > 0) & (missing_pct <= 0.3)].index

        # 3) IterativeImputer for "high_missing" columns
        if len(high_missing) > 0:
            imp_iter = IterativeImputer(random_state=42)
            data[high_missing] = imp_iter.fit_transform(data[high_missing])

        # 4) KNNImputer for "low_missing" columns
        if len(low_missing) > 0:
            imp_knn = KNNImputer(n_neighbors=5)
            data[low_missing] = imp_knn.fit_transform(data[low_missing])

        # 5) Final forward/backward fill for any remaining holes
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using RobustScaler for columns: Open, High, Low, Close, Volume.
        Outliers beyond threshold get turned into NaN, then re-imputed.
        """
        data = data.copy()
        scaler = RobustScaler()
        cols_to_scale = ["Open", "High", "Low", "Close", "Volume"]

        for col in cols_to_scale:
            if col not in data.columns:
                continue
            # Convert to numeric to avoid errors
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Apply robust scaling
            scaled = scaler.fit_transform(data[col].values.reshape(-1, 1))
            # Mark outliers
            mask_outliers = (abs(scaled) > 3)  # threshold => 3
            data.loc[mask_outliers.ravel(), col] = np.nan

        # Re-impute missing if any outliers were set to NaN
        data = self._handle_missing_values(data)
        return data

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new features from cleaned data using FeatureEngineer.
        """
        return self.feature_engineer.process_features(data)
