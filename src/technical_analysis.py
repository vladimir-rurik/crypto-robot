"""
technical_analysis.py

Demonstrates basic technical analysis on a wide DataFrame,
computing RSI, MACD, Bollinger Bands, and EMA for each symbol,
while avoiding DataFrame fragmentation warnings by concatenating once.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # pip install ta


def main():
    # 1) Load combined data (long format) and pivot to wide
    csv_path = os.path.join("data", "combined_minute_data.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found at {csv_path}.")
        return

    print(f"[INFO] Loading data from {csv_path}")
    df_long = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print("[INFO] Pivoting data to wide format...")

    df_wide = df_long.pivot(index="timestamp", columns="symbol", values="Close")
    df_wide.dropna(how="all", inplace=True)

    symbols = list(df_wide.columns)
    print(f"[INFO] Found symbols: {symbols}")

    # 2) Collect each symbol's TA columns into a list of dataframes
    df_list = []

    for symbol in symbols:
        # Extract that symbol's close price
        close_series = df_wide[symbol].ffill().bfill().dropna()

        # Calculate indicators using 'ta' library
        rsi_indicator = ta.momentum.RSIIndicator(close_series, fillna=True)
        macd_indicator = ta.trend.MACD(close_series, fillna=True)
        bb_indicator = ta.volatility.BollingerBands(close_series, fillna=True)
        ema_indicator = ta.trend.EMAIndicator(close_series, window=50, fillna=True)

        # Build a dictionary of new columns
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

        # Create a temporary DataFrame
        temp_df = pd.DataFrame(data_dict, index=close_series.index)

        # Add to list
        df_list.append(temp_df)

    # 3) Concatenate all symbols' DataFrames side-by-side in one operation
    df_ta = pd.concat(df_list, axis=1)

    print("\n[INFO] Technical indicators computed.")
    print("[INFO] df_ta sample:\n", df_ta.head())

    # 4) Example: plot for one symbol
    example_symbol = "BTC"
    if example_symbol in symbols:
        plot_symbol_ta(df_wide, df_ta, example_symbol)
    else:
        print(f"[WARNING] {example_symbol} not found in data. Skipping plot.")

    # 5) Save the final DataFrame
    out_path = os.path.join("data", "technical_indicators.csv")
    df_ta.to_csv(out_path)
    print(f"[INFO] Full technical indicators saved to {out_path}")
    print("[INFO] Done.")


def plot_symbol_ta(df_wide, df_ta, symbol):
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax_price, ax_rsi, ax_macd = axes

    ax_price.set_title(f"{symbol} Price + Bollinger(20) + EMA(50)")
    ax_price.plot(df_wide.index, df_wide[symbol], label=f"{symbol} Close", color="black", linewidth=1)
    ax_price.plot(df_ta[f"{symbol}_BB_high"], label="BB High", color="red", linewidth=0.8)
    ax_price.plot(df_ta[f"{symbol}_BB_low"], label="BB Low", color="blue", linewidth=0.8)
    ax_price.plot(df_ta[f"{symbol}_EMA50"], label="EMA50", color="magenta", linewidth=0.8)
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.3)

    ax_rsi.set_title(f"{symbol} RSI")
    ax_rsi.plot(df_ta[f"{symbol}_RSI"], color="green", linewidth=1)
    ax_rsi.axhline(30, color="red", linestyle="--", alpha=0.5)
    ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax_rsi.grid(True, alpha=0.3)

    ax_macd.set_title(f"{symbol} MACD")
    ax_macd.plot(df_ta[f"{symbol}_MACD"], color="blue", label="MACD")
    ax_macd.plot(df_ta[f"{symbol}_MACD_signal"], color="orange", label="Signal")
    ax_macd.bar(df_ta.index, df_ta[f"{symbol}_MACD_diff"], color="gray", label="MACD Diff", width=0.0001)
    ax_macd.legend(loc="upper left")
    ax_macd.grid(True, alpha=0.3)

    # Format time axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    plt.savefig("tech_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
