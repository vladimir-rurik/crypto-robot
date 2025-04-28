"""
technical_analysis.py
---------------------
Compute technical indicators (RSI, MACD, Bollinger Bands, EMA) for *all* symbols
found in combined_minute_data.csv. Writes out a single CSV file (technical_indicator.csv)
with columns for every coin’s indicators.

Requires: pip install ta matplotlib pandas numpy
"""

import os
import pandas as pd
import numpy as np
import ta  # library for technical indicators


def main():
    # 1) Load long-format data (symbol, timestamp, Close)
    csv_path = os.path.join("data", "combined_minute_data.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found at {csv_path}.")
        return

    print(f"[INFO] Loading data from {csv_path}")
    df_long = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # 2) Pivot to wide: each symbol => a separate 'Close' column
    print("[INFO] Pivoting data to wide format (Close prices)...")
    df_wide = df_long.pivot(index="timestamp", columns="symbol", values="Close")

    # (Optional) Drop rows fully NaN
    df_wide.dropna(how="all", inplace=True)

    symbols = df_wide.columns.tolist()
    print(f"[INFO] Found {len(symbols)} symbols: {symbols}")

    # 3) Collect per-symbol indicator data in a list to avoid repeated DataFrame inserts
    df_list = []

    # For each symbol, compute RSI, MACD, Bollinger, EMA, etc.
    for symbol in symbols:
        # Extract that symbol’s close series
        close_series = df_wide[symbol].ffill().bfill().dropna()

        # Skip symbols that have too little data
        if len(close_series) < 10:
            print(f"[WARNING] {symbol} has too few data points; skipping.")
            continue

        # Create indicators using ta
        rsi_indicator = ta.momentum.RSIIndicator(close_series, fillna=True)
        macd_indicator = ta.trend.MACD(close_series, fillna=True)
        bb_indicator = ta.volatility.BollingerBands(close_series, fillna=True)
        ema_indicator = ta.trend.EMAIndicator(close_series, window=50, fillna=True)

        # Build a dictionary of new columns for this symbol
        data_dict = {
            f"{symbol}_RSI": rsi_indicator.rsi(),
            f"{symbol}_MACD": macd_indicator.macd(),
            f"{symbol}_MACD_signal": macd_indicator.macd_signal(),
            f"{symbol}_MACD_diff": macd_indicator.macd_diff(),
            f"{symbol}_BB_high": bb_indicator.bollinger_hband(),
            f"{symbol}_BB_low": bb_indicator.bollinger_lband(),
            f"{symbol}_BB_mavg": bb_indicator.bollinger_mavg(),
            f"{symbol}_EMA50": ema_indicator.ema_indicator(),
        }

        # Make a small DataFrame indexed by the same timestamps
        temp_df = pd.DataFrame(data_dict, index=close_series.index)
        df_list.append(temp_df)

    # 4) Concatenate all symbols' indicator DataFrames side-by-side
    df_ta = pd.concat(df_list, axis=1).sort_index()

    # 5) Optionally fill remaining NaNs or keep them
    df_ta.ffill(inplace=True)
    df_ta.bfill(inplace=True)

    print("\n[INFO] Example of df_ta:\n", df_ta.head())

    # 6) Write final CSV with all coins' technical indicators
    out_path = os.path.join("data", "technical_indicator.csv")
    df_ta.to_csv(out_path)
    print(f"[INFO] Technical indicators for all coins saved to: {out_path}")


if __name__ == "__main__":
    main()
