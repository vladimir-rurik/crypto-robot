"""
Data processing and feature engineering pipeline script.
Main script for running data preprocessing, feature engineering,
and consolidating data into a single CSV for further modeling.
No database creation or usage.
"""

import os
import pandas as pd

from market_analyzer.preprocessor import DataPreprocessor
from market_analyzer.data_dashboard import DataProcessingDashboard  # optional for visualization

# list of symbols
CRYPTO_SYMBOLS = [
    "ETH", "SOL", "ADA", "LINK", "AVAX",
    "XLM", "LTC", "DOT", "UNI", "AAVE",
    "SAND", "AXS", "MATIC", "FTM", "BTC"
]

def main():
    """Run data processing pipeline using local minutely CSV files only."""
    os.makedirs('data', exist_ok=True)
    
    # Initialize our data preprocessor (no db_path needed)
    preprocessor = DataPreprocessor()
    dashboard = DataProcessingDashboard()  # optional if you want data quality plots

    all_processed = []

    for symbol in CRYPTO_SYMBOLS:
        # Construct the path to the CSV file
        file_path = os.path.join('..\data', f"{symbol}_minutely_data.csv")
        if not os.path.exists(file_path):
            print(f"[Warning] File not found for {symbol}: {file_path}")
            continue

        print(f"\nLoading {symbol} from {file_path} ...")
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')

        # Rename columns if your CSV differs
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)

        # Clean the data
        cleaned_data = preprocessor.clean_data(df)

        # Feature engineering
        features = preprocessor.engineer_features(cleaned_data)

        # Combine cleaned columns with engineered features
        combined = cleaned_data.join(features, how="inner")
        combined['symbol'] = symbol

        # (Optional) Dashboard: data quality, summary plots, etc.
        # dashboard.plot_data_quality(cleaned_data)
        # dashboard.plot_summary_dashboard(cleaned_data, features)

        all_processed.append(combined)

    # Merge all symbols into a single DataFrame and write to CSV
    if all_processed:
        final_df = pd.concat(all_processed, axis=0)
        final_df.sort_index(inplace=True)
        out_csv = os.path.join('data', 'combined_minute_data.csv')
        final_df.to_csv(out_csv)
        print(f"\nAll symbols processed successfully.")
        print(f"Combined data written to => {out_csv}")
    else:
        print("No data processed. Check if CSV files exist.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
    finally:
        print("\nData processing pipeline finished.")
