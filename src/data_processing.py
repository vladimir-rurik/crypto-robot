import os
import time
import pandas as pd

from market_analyzer.preprocessor import DataPreprocessor
from market_analyzer.data_dashboard import DataProcessingDashboard

CRYPTO_SYMBOLS = [
    "ETH", "SOL", "ADA", "LINK", "AVAX",
    "XLM", "LTC", "DOT", "UNI", "AAVE",
    "SAND", "AXS", "MATIC", "FTM", "BTC"
]

def main():
    print("[INFO] Starting data processing pipeline...")
    os.makedirs('data', exist_ok=True)
    
    preprocessor = DataPreprocessor()
    dashboard = DataProcessingDashboard()  # optional
    
    all_processed = []
    n_symbols_processed = 0

    for symbol in CRYPTO_SYMBOLS:
        file_path = os.path.join('..', 'data', f"{symbol}_minutely_data.csv")
        if not os.path.exists(file_path):
            print(f"[WARNING] CSV not found for {symbol} => {file_path}")
            continue

        print(f"\n[INFO] Loading {symbol} from {file_path} ...")
        start_time = time.time()

        # Read CSV
        df = pd.read_csv(
            file_path,
            parse_dates=["timestamp"],  # Adjust to your actual datetime column
            index_col="timestamp"
        )
        # Rename columns
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)

        # Clean data
        print(f"[DEBUG]  -> Cleaning {symbol} data ...")
        cleaned_data = preprocessor.clean_data(df)

        # Feature engineering
        print(f"[DEBUG]  -> Engineering features for {symbol}...")
        features = preprocessor.engineer_features(cleaned_data)

        # Combine
        combined = cleaned_data.join(features, how="inner")
        combined["symbol"] = symbol

        all_processed.append(combined)
        n_symbols_processed += 1
        
        # Print how long it took for this symbol
        elapsed = time.time() - start_time
        print(f"[INFO] Done processing {symbol} in {elapsed:.2f} seconds.")

    # If any symbols were processed, save combined
    if all_processed:
        print(f"\n[INFO] Concatenating data for {n_symbols_processed} symbols...")
        final_df = pd.concat(all_processed, axis=0)
        final_df.sort_index(inplace=True)

        out_csv = os.path.join('..', 'data', 'combined_minute_data.csv')
        final_df.to_csv(out_csv)
        print(f"[INFO] Combined data written to => {out_csv}")
    else:
        print("[WARNING] No data processed. Check your CSV files.")

    print("[INFO] Data processing pipeline finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {str(e)}")
