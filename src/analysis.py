"""
analysis_lag.py
Deep dive: Loop over all crypto symbol pairs, find best cross-correlation (lead-lag).
Run with:  python analysis_lag.py
"""

import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

def main():
    # --------------------------------------------------
    # 1) LOAD AND PIVOT THE DATA
    # --------------------------------------------------
    csv_path = os.path.join("data", "combined_minute_data.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    print(f"[INFO] Loading data from: {csv_path}")
    df_long = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Pivot to wide: each symbol in a separate column for 'Close'
    df_wide = df_long.pivot(index="timestamp", columns="symbol", values="Close")

    # Drop any rows that are all-NaN
    df_wide.dropna(how="all", inplace=True)

    symbols = df_wide.columns.tolist()
    print(f"[INFO] Found {len(symbols)} symbols: {symbols}")

    # --------------------------------------------------
    # 2) LOOP OVER ALL PAIRS, FIND BEST LAG
    # --------------------------------------------------
    max_lag = 30  # +/- 30 minutes
    results = []  # will hold dicts of {s1, s2, best_lag, best_corr}

    # Generate all unique pairs (s1, s2), s1 != s2
    for s1, s2 in itertools.permutations(symbols, 2):
        # For each pair, compute best correlation across lags
        best_corr, best_lag = find_best_lag(df_wide[s1], df_wide[s2], max_lag)
        # Store result
        results.append({
            "symbol_x": s1,
            "symbol_y": s2,
            "best_lag": best_lag,
            "best_corr": best_corr
        })
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    # Sort by absolute correlation descending (or just positive correlation)
    df_results.sort_values(by="best_corr", ascending=False, inplace=True)

    # Print top 20
    print("\n[INFO] Top 20 pairs by correlation magnitude:")
    print(df_results.head(20).round(3))

    # --------------------------------------------------
    # 3) OPTIONAL: PLOT SOME OF THESE PAIRS
    # --------------------------------------------------
    # For example, take the top 1 pair
    if not df_results.empty:
        top_pair = df_results.iloc[0]
        s1, s2, lag, corr = top_pair["symbol_x"], top_pair["symbol_y"], top_pair["best_lag"], top_pair["best_corr"]
        print(f"\n[INFO] Highest correlation pair: {s1} vs. {s2} => Corr={corr:.3f} @ lag={lag}")

        # Plot cross-correlation for that pair specifically
        cross_corr_plot(df_wide[s1], df_wide[s2], max_lag=max_lag)

    print("\n[INFO] analysis_lag.py complete.")

def find_best_lag(series_x, series_y, max_lag=30):
    """
    For the given series_x, series_y, compute correlation at lags in [-max_lag..+max_lag].
    Return (best_corr, best_lag) where best_corr is the maximum correlation found.
    """
    # Align & drop NaN
    df = pd.DataFrame({"X": series_x, "Y": series_y}).dropna()
    best_corr = float("-inf")
    best_lag = 0

    for lag in range(-max_lag, max_lag+1):
        # Shift Y by 'lag'
        shifted_y = df["Y"].shift(lag)
        corr_val = df["X"].corr(shifted_y)
        if corr_val is not None and corr_val > best_corr:
            best_corr = corr_val
            best_lag = lag

    return best_corr, best_lag

def cross_corr_plot(series_x, series_y, max_lag=30):
    """
    Plot correlation vs. lag. A positive lag means 'series_y' is shifted forward.
    """
    df = pd.DataFrame({"X": series_x, "Y": series_y}).dropna()
    results = {}

    for lag in range(-max_lag, max_lag+1):
        shifted_y = df["Y"].shift(lag)
        corr_val = df["X"].corr(shifted_y)
        results[lag] = corr_val

    lags = list(results.keys())
    corrs = list(results.values())

    # Find best
    best_lag = max(results, key=results.get)
    best_corr = results[best_lag]

    plt.figure(figsize=(10, 5))
    plt.axhline(y=0, color="gray", linestyle="--")
    plt.plot(lags, corrs, marker="o")
    plt.title(f"Cross-correlation: X={series_x.name} vs. Y={series_y.name}\nBest Corr={best_corr:.3f} @ lag={best_lag}")
    plt.xlabel("Lag (minutes) [Y shifted]")
    plt.ylabel("Pearson correlation")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
