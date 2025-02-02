"""
analysis.py
A script to analyze lead-lag correlations and seasonality
in the combined_minute_data.csv dataset.

Run:  python analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# If you want seasonal_decompose:
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARNING] statsmodels not installed; skipping seasonal decomposition.")


def main():
    # 1) Load combined CSV
    csv_path = os.path.join('..', 'data', 'combined_minute_data.csv')
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    print(f"[INFO] Loading combined data from: {csv_path}")
    df_long = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # 2) Pivot into wide format (Close price by symbol)
    print("[INFO] Pivoting data to wide format ...")
    if "symbol" not in df_long.columns:
        print("[ERROR] 'symbol' column is missing in combined CSV.")
        return
    if "Close" not in df_long.columns:
        print("[ERROR] 'Close' column is missing in combined CSV.")
        return

    df_wide = df_long.pivot(index="timestamp", columns="symbol", values="Close")

    # 3) Basic correlation matrix (zero-lag)
    corr_matrix = df_wide.corr()
    print("\n[INFO] Zero-lag Correlation Matrix (Close prices):")
    print(corr_matrix.round(3))

    # 4) Example: cross-correlation between two chosen cryptos
    symbol_x = "BTC"  # Adjust to your actual symbols in the CSV
    symbol_y = "ETH"

    if symbol_x not in df_wide.columns or symbol_y not in df_wide.columns:
        print(f"[WARNING] {symbol_x} or {symbol_y} not found in columns.")
    else:
        print(f"\n[INFO] Performing cross-correlation: {symbol_x} vs. {symbol_y}")
        cross_corr_plot(df_wide[symbol_x], df_wide[symbol_y], max_lag=30)

    # 5) Optionally see seasonality in a single crypto (e.g. BTC)
    if HAS_STATSMODELS and "BTC" in df_wide.columns:
        print("[INFO] Doing seasonal decomposition for BTC ...")
        seasonal_decomp_plot(df_wide["BTC"], period=1440)  # 1440 for daily pattern (minutes)

    print("\n[INFO] Analysis complete.")


def cross_corr_plot(series_x, series_y, max_lag=30):
    """
    Compute and plot cross-correlation for lags from -max_lag..+max_lag.
    A positive lag means 'series_y' is shifted forward.
    """
    # Ensure both are properly aligned by time
    df = pd.DataFrame({'X': series_x, 'Y': series_y}).dropna()

    results = {}
    for lag in range(-max_lag, max_lag+1):
        # shift Y by 'lag'
        shifted_y = df['Y'].shift(lag)
        corr_val = df['X'].corr(shifted_y)
        results[lag] = corr_val

    # Plot
    lags = list(results.keys())
    corrs = list(results.values())

    plt.figure(figsize=(10, 5))
    plt.axhline(0, color='gray', linestyle='--')
    plt.plot(lags, corrs, marker='o')
    plt.title(f"Cross-correlation: X={series_x.name} vs Y={series_y.name}")
    plt.xlabel("Lag (minutes) [Y shifted]")
    plt.ylabel("Pearson correlation")
    plt.grid(True, alpha=0.3)

    # Mark the max correlation
    best_lag = max(results, key=results.get)
    best_corr = results[best_lag]
    plt.annotate(f"Max Corr {best_corr:.3f} @ lag={best_lag}",
                 xy=(best_lag, best_corr),
                 xytext=(best_lag, best_corr+0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    plt.show()


def seasonal_decomp_plot(series, period=1440):
    """
    Attempt a statsmodels seasonal decomposition plot.
    'series' should be a regular time-series, e.g. minutely data.
    'period=1440' for daily pattern if it's 1-min frequency data.
    """
    # Make sure 'series' index is datetime and has a freq:
    series = series.asfreq('T')  # 'T' is minute freq
    series = series.fillna(method='ffill')

    result = seasonal_decompose(series, period=period, model='additive')
    fig = result.plot()
    fig.suptitle(f"Seasonal Decomposition: {series.name}", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
