"""
Core analyzer module for financial market data analysis.
usage—only local CSV files.
"""

import os
import pandas as pd
from typing import Dict, List, Optional


class MarketDataAnalyzer:
    """
    A class for analyzing financial market data locally.
    It reads CSV files (minutely or otherwise) from a specified directory
    and stores them in memory for further use.
    """

    def __init__(
        self,
        data_dir: str = "data",
        crypto_symbols: Optional[List[str]] = None
    ):
        """
        Initialize the analyzer with local CSV file paths.

        Args:
            data_dir: Directory where your CSV files are stored.
            crypto_symbols: List of crypto symbols to load; 
                            if None, uses a default set.
        """
        if crypto_symbols is None:
            # Provide the symbols you want to track
            # (Example: the same list used in your data_processing pipeline)
            crypto_symbols = [
                "BTC", "ETH", "SOL", "ADA", "LINK",
                "AVAX", "XLM", "LTC", "DOT", "UNI",
                "AAVE", "SAND", "AXS", "MATIC", "FTM"
            ]
        self.data_dir = data_dir
        self.crypto_symbols = crypto_symbols

        # In-memory storage
        self.crypto_data: Dict[str, pd.DataFrame] = {}

    def load_local_data(self) -> None:
        """
        Load local CSV files (e.g. minutely data) for each symbol in self.crypto_symbols.
        Adjust column names/index parsing as needed.
        """
        print("Loading local CSV data for all crypto symbols...")

        for symbol in self.crypto_symbols:
            file_name = f"{symbol}_minutely_data.csv"  # or your preferred naming
            path = os.path.join(self.data_dir, file_name)

            if not os.path.exists(path):
                print(f"  [Warning] File not found: {path}")
                continue

            print(f"  Reading {symbol} data from {path}")
            df = pd.read_csv(path, parse_dates=["time"], index_col="time")

            # Rename columns to match the rest of your pipeline if needed
            # For example:
            df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }, inplace=True)

            # Optional data preparation/cleanup
            df = self._prepare_data(df)

            # Store in dictionary
            self.crypto_data[symbol] = df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare local CSV data by handling missing values, ensuring
        correct datetime index, etc.

        Args:
            df: Raw DataFrame read from CSV.

        Returns:
            Processed DataFrame, cleaned and ready for analysis.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # Make sure index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Convert numeric columns
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Forward-fill/back-fill any NaNs (simple approach—adjust as needed)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

        return df

    def check_data_quality(self) -> Dict[str, Dict]:
        """
        Check data quality for all loaded assets, returning
        a summary dictionary.

        Returns:
            Dictionary keyed by symbol, with basic data health stats.
        """
        quality_report = {}

        def check_asset(asset_df: pd.DataFrame) -> Dict:
            if asset_df.empty:
                return {
                    "missing_values": {},
                    "start_date": None,
                    "end_date": None,
                    "total_rows": 0,
                    "unique_dates": 0,
                }
            return {
                "missing_values": asset_df.isnull().sum().to_dict(),
                "start_date": asset_df.index.min(),
                "end_date": asset_df.index.max(),
                "total_rows": len(asset_df),
                "unique_dates": asset_df.index.nunique(),
            }

        for symbol, data in self.crypto_data.items():
            quality_report[symbol] = check_asset(data)

        return quality_report

    def get_asset_data(self, symbol: str) -> pd.DataFrame:
        """
        Get data for a specific symbol that was loaded.

        Args:
            symbol: The crypto symbol to retrieve.

        Returns:
            DataFrame containing that symbol’s DataFrame.
        """
        if symbol in self.crypto_data:
            return self.crypto_data[symbol]
        else:
            raise ValueError(f"Symbol {symbol} not found in loaded data.")
