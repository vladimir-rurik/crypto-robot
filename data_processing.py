"""
Data processing and feature engineering pipeline script.
Main script for running data preprocessing, feature engineering,
and consolidating data into a single CSV for further ensemble modeling.
"""

import os
import glob
import pandas as pd

from market_analyzer.preprocessor import DataPreprocessor
from market_analyzer.data_dashboard import DataProcessingDashboard


# List only the crypto symbols from your attached list
CRYPTO_SYMBOLS = [
    "ETH", "SOL", "ADA", "LINK", "AVAX",
    "XLM", "LTC", "DOT", "UNI", "AAVE",
    "SAND", "AXS", "MATIC", "FTM"
]


def main():
    """Run data processing pipeline using local minutely CSV files."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    print("Initializing data processing pipeline...")
    preprocessor = DataPreprocessor(db_path='data/market_data.db')
    dashboard = DataProcessingDashboard()

    # This DataFrame will hold all cleaned+engineered rows from all symbols
    all_processed = []

    # -------------------------------------------------------------------------
    # Read each symbol's minutely CSV, process, and append to a single DataFrame
    # -------------------------------------------------------------------------
    for symbol in CRYPTO_SYMBOLS:
        # Construct file path, e.g. "data/ETH_minutely_data.csv"
        file_path = os.path.join('data', f"{symbol}_minutely_data.csv")
        if not os.path.exists(file_path):
            print(f"File not found for {symbol}: {file_path}")
            continue

        print(f"\nLoading {symbol} from {file_path} ...")
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

        # Optional: Rename columns if needed to match your pipeline
        # e.g., df.rename(columns={"open":"Open","high":"High",...}, inplace=True)
        # Adjust to match the exact columns your pipeline expects:
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        # Preprocessing steps
        print("  Cleaning data...")
        cleaned_data = preprocessor.clean_data(df)

        print("  Feature engineering...")
        features = preprocessor.engineer_features(cleaned_data)

        # If you still want to store into SQLite, call process_new_data:
        # preprocessor.process_new_data(symbol, cleaned_data)

        # Combine cleaned (price) + features side by side:
        combined = cleaned_data.join(features, how="inner")
        combined['symbol'] = symbol  # Label with the symbol

        # You can do any final checks or dashboards here
        # (Plots, distributions, etc.) for each symbol if you like
        # dashboard.plot_data_quality(cleaned_data)
        # dashboard.plot_summary_dashboard(cleaned_data, features)

        # Append to the big list
        all_processed.append(combined)

    # Concatenate all symbols into one final DataFrame
    if all_processed:
        final_df = pd.concat(all_processed, axis=0)
        final_df.sort_index(inplace=True)

        # Write to a single CSV for further modeling/ensembles
        out_csv = os.path.join('data', 'combined_minute_data.csv')
        final_df.to_csv(out_csv, index=True)
        print(f"\nAll symbols processed successfully.")
        print(f"Combined data written to: {out_csv}")
    else:
        print("No data processed. Check if your minutely CSV files exist.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
    finally:
        print("\nData processing pipeline finished.")
