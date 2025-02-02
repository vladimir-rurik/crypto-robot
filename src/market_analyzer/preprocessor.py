"""
Data preprocessing and feature engineering pipeline.
Handles cleaning, missing data, outliers, and calls FeatureEngineer.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from .feature_engineering import FeatureEngineer
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

class DataPreprocessor:
    def __init__(self, db_path='data/market_data.db'):
        self.db_path = db_path
        self.feature_engineer = FeatureEngineer()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.initialize_database()

    def initialize_database(self):
        # Creates DB tables if you want to store raw/processed data
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    date TEXT,
                    symbol TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (date, symbol)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processed_data (
                    date TEXT,
                    symbol TEXT,
                    feature_name TEXT,
                    value REAL,
                    PRIMARY KEY (date, symbol, feature_name)
                )
            ''')

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # 1) Handle missing data
        data = self._handle_missing_values(data)
        # 2) Remove outliers using robust methods
        data = self._remove_outliers(data)
        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        missing_pct = data.isnull().sum() / len(data)

        # If entire column is missing, fill with synthetic data
        all_missing = missing_pct[missing_pct == 1.0].index
        for col in all_missing:
            # Some logic to fill in synthetic values
            data[col] = np.random.normal(data.mean().mean(), data.std().mean(), len(data))

        # Splitting columns by missing % for different imputation strategies
        high_missing = missing_pct[(missing_pct > 0.3) & (missing_pct < 1.0)].index
        low_missing  = missing_pct[missing_pct <= 0.3].index

        if len(high_missing) > 0:
            imp = IterativeImputer(random_state=42)
            data[high_missing] = imp.fit_transform(data[high_missing])

        if len(low_missing) > 0:
            imp = KNNImputer()
            data[low_missing] = imp.fit_transform(data[low_missing])

        # Final forward/backward fill
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        scaler = RobustScaler()
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        for col in cols:
            if col not in data.columns: 
                continue
            col_vals = data[col].values.reshape(-1, 1)
            scaled = scaler.fit_transform(col_vals)
            # Mask out anything above threshold
            outlier_mask = np.abs(scaled) > 3
            data.loc[outlier_mask.ravel(), col] = np.nan

        # Re-impute after removing outliers
        data = self._handle_missing_values(data)
        return data

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engineer.process_features(data)

    def process_new_data(self, symbol: str, new_data: pd.DataFrame):
        # 1) Clean data
        cleaned_data = self.clean_data(new_data)
        # 2) Feature engineering
        features = self.engineer_features(cleaned_data)

        # 3) Store data in DB
        with sqlite3.connect(self.db_path) as conn:
            for idx, row in cleaned_data.iterrows():
                conn.execute('''
                    INSERT OR IGNORE INTO raw_data
                    (date, symbol, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (idx.strftime('%Y-%m-%d %H:%M:%S'), symbol,
                      row['Open'], row['High'], row['Low'],
                      row['Close'], row['Volume']))
            for col in features.columns:
                for idx, val in features[col].items():
                    if pd.notna(val):
                        conn.execute('''
                            INSERT OR IGNORE INTO processed_data
                            (date, symbol, feature_name, value)
                            VALUES (?, ?, ?, ?)
                        ''', (idx.strftime('%Y-%m-%d %H:%M:%S'),
                              symbol, col, val))

    def get_latest_data(self, symbol: str,
                        start_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Retrieve from DB if needed
        ...
