"""
analysis_ratio.py
Check if one crypto price is "usually bigger" than another's, 
and create a full table of all pairs (X, Y).

Run:  python analysis_ratio.py
"""

import os
import pandas as pd
import numpy as np
import itertools

def main():
    # ---------------------------------------
    # 1) LOAD AND PIVOT THE DATA
    # ---------------------------------------
    csv_path = os.path.join("data", "combined_minute_data.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    print(f"[INFO] Loading data from: {csv_path}")
    df_long = pd.read_csv(csv_path, parse_dates=["timestamp"])
    
    # Pivot to wide: each symbol in a separate column for "Close" price
    df_wide = df_long.pivot(index="timestamp", columns="symbol", values="Close")
    df_wide.dropna(how="all", inplace=True)  # remove rows entirely NaN

    symbols = df_wide.columns.tolist()
    print(f"[INFO] Symbols found: {symbols}")

    # ---------------------------------------
    # 2) CALCULATE RATIO X/Y FOR ALL PAIRS
    # ---------------------------------------
    results = []

    # permutations(...) => includes (A, B) and (B, A)
    for X, Y in itertools.permutations(symbols, 2):
        pair_df = df_wide[[X, Y]].dropna()
        # ratio = X / Y
        pair_df["ratio"] = pair_df[X] / pair_df[Y]

        mean_ratio = pair_df["ratio"].mean()
        std_ratio = pair_df["ratio"].std()

        # Fraction of time ratio < 1 => "inversion"
        fraction_inverted = (pair_df["ratio"] < 1).mean()

        results.append({
            "symbol_x": X,
            "symbol_y": Y,
            "mean_ratio": mean_ratio,
            "std_ratio": std_ratio,
            "pct_inversion": fraction_inverted
        })

    df_results = pd.DataFrame(results)

    # Optional: round for cleaner display
    df_results = df_results.round(4)

    # ---------------------------------------
    # 3) PRINT OR SAVE THE FULL TABLE
    # ---------------------------------------
    print("\n[INFO] FULL TABLE of all pairs (X, Y) with ratio X/Y:\n")
    # Print entire DataFrame to screen:
    print(df_results.to_string(index=False))

    # If you want to export this entire table to a CSV:
    out_csv = os.path.join("data", "all_pairs_ratios.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n[INFO] Full pairs table saved to => {out_csv}")

    # ---------------------------------------
    # 4) PIVOT FOR A MATRIX VIEW (MEAN_RATIO)
    # ---------------------------------------
    # Create a matrix with rows=X, columns=Y, cell = mean_ratio
    ratio_matrix = df_results.pivot(
        index="symbol_x",
        columns="symbol_y",
        values="mean_ratio"
    )

    print("\n[INFO] Matrix of mean_ratio (X / Y) for each pair:")
    # Round for display
    print(ratio_matrix.round(2).fillna("N/A"))

    # You could similarly pivot for std_ratio or pct_inversion if desired.
    # e.g.:
    # pivot_inversion = df_results.pivot(
    #     index="symbol_x",
    #     columns="symbol_y",
    #     values="pct_inversion"
    # )
    # print("\n[INFO] Matrix of pct_inversion (X / Y < 1):")
    # print(pivot_inversion.round(2).fillna("N/A"))

    print("\n[INFO] analysis_ratio.py complete.")

if __name__ == "__main__":
    main()
