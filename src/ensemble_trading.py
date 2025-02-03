"""
ensemble_trading.py
-------------------
A simplified demo of building multiple ML models to generate trading signals,
then backtesting vs. a buy-and-hold baseline.
"""

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Train/test splitting
from sklearn.model_selection import train_test_split

# For performance metrics
from sklearn.metrics import accuracy_score

def main():
    # -----------------------------------------------------
    # 1) LOAD AND PREPARE DATA
    # -----------------------------------------------------
    csv_path = os.path.join("data", "technical_indicator.csv")  
    if not os.path.exists(csv_path):
        print(f"[ERROR] No CSV found at {csv_path}")
        return

    print(f"[INFO] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    # The DataFrame columns might look like: SOL_RSI, SOL_MACD, etc. 
    # For this example, let's pick one symbol, e.g., "SOL" 
    # and use some columns as features:
    feature_cols = [
        "SOL_RSI",
        "SOL_MACD",
        "SOL_MACD_signal",
        "SOL_MACD_diff",
        "SOL_BB_high",
        "SOL_BB_low",
        "SOL_BB_mavg",
        "SOL_EMA50",
    ]
    # We'll assume these columns exist in df; 
    # fill or drop missing
    df = df[feature_cols].copy().fillna(method="ffill").dropna()

    # We also need a TARGET (e.g., future 1-period price move).
    # Typically we read from the original close price or a "future_return" column in your data.
    # For demonstration, let's say you already have df["SOL_Close"] in a separate file,
    # or we can re-load from the combined data if needed.

    # (Here we'll do a quick hack: we re-load the combined data for the SOL Close price.)
    combined_csv = os.path.join("data", "combined_minute_data.csv")
    if not os.path.exists(combined_csv):
        print("[ERROR] Missing combined data for price reference. Quitting.")
        return
    df_price = pd.read_csv(combined_csv, parse_dates=["timestamp"], index_col="timestamp")
    # pivot to wide, grab 'SOL' column
    df_sol = df_price.pivot(columns="symbol", values="Close")
    df_sol = df_sol["SOL"].rename("SOL_Close").ffill().dropna()

    # align with our feature DataFrame
    df = df.join(df_sol, how="inner")

    # Next, define a binary classification target: e.g. +1 if next close > current close, else 0
    df["future_close"] = df["SOL_Close"].shift(-1)  # next period's close
    df["target"] = np.where(df["future_close"] > df["SOL_Close"], 1, 0)
    df.dropna(inplace=True)

    print("[INFO] Final shape of data with target:", df.shape)

    # -----------------------------------------------------
    # 2) SPLIT INTO TRAIN/TEST
    # -----------------------------------------------------
    X = df[feature_cols]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")

    # -----------------------------------------------------
    # 3) TRAIN MULTIPLE MODELS
    # -----------------------------------------------------
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric="logloss")
    lr = LogisticRegression()

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # Predict on test set
    pred_rf = rf.predict(X_test)
    pred_xgb = xgb.predict(X_test)
    pred_lr = lr.predict(X_test)

    # Evaluate basic accuracy
    print("[INFO] Accuracies:")
    print("  RF :", accuracy_score(y_test, pred_rf))
    print("  XGB:", accuracy_score(y_test, pred_xgb))
    print("  LR :", accuracy_score(y_test, pred_lr))

    # -----------------------------------------------------
    # 4) BUILD ENSEMBLE SIGNAL
    # -----------------------------------------------------
    # Simple majority vote: e.g., 2-of-3 must agree to predict 1
    pred_ensemble = (pred_rf + pred_xgb + pred_lr)  # sums each row's predictions
    # If sum >= 2, final pred = 1; else = 0
    pred_ensemble = np.where(pred_ensemble >= 2, 1, 0)

    ensemble_acc = accuracy_score(y_test, pred_ensemble)
    print("[INFO] Ensemble accuracy:", ensemble_acc)

    # We'll create a small DataFrame that holds the model signals:
    df_signals = pd.DataFrame(
        index=X_test.index,
        data={
            "RF": pred_rf,
            "XGB": pred_xgb,
            "LR": pred_lr,
            "Ensemble": pred_ensemble
        },
    )
    # '1' means "predict price up => BUY," '0' means "predict price down => SELL or do nothing."

    # -----------------------------------------------------
    # 5) BACKTEST THE ENSEMBLE STRATEGY
    # -----------------------------------------------------
    # For a basic demonstration, let's do a very naive backtest:
    # - We'll take the ensemble's predicted action as: +1 (long) if ensemble=1, or 0 (flat) if ensemble=0
    # - Then compute hypothetical returns.
    # - Compare to buy & hold returns.

    df_backtest = X_test.copy()
    df_backtest["close"] = df.loc[X_test.index, "SOL_Close"]
    df_backtest["ensemble_signal"] = df_signals["Ensemble"]

    # Shift signals so we "enter" at next bar open
    df_backtest["ensemble_signal"] = df_backtest["ensemble_signal"].shift(1).fillna(0)

    # Compute daily (or in this case, minute) returns
    df_backtest["pct_change"] = df_backtest["close"].pct_change().fillna(0)

    # Strategy returns = signal * market returns
    # (If signal=1 => we are long, so we get that day's return, else 0 => no position.)
    df_backtest["strategy_return"] = df_backtest["ensemble_signal"] * df_backtest["pct_change"]

    # Buy & hold baseline:
    df_backtest["bh_return"] = df_backtest["pct_change"]  # always in market

    # Compute cumulative returns
    df_backtest["cum_strategy"] = (1 + df_backtest["strategy_return"]).cumprod()
    df_backtest["cum_bh"] = (1 + df_backtest["bh_return"]).cumprod()

    final_strat = df_backtest["cum_strategy"].iloc[-1] - 1.0
    final_bh = df_backtest["cum_bh"].iloc[-1] - 1.0

    print("\n[INFO] Final results:")
    print(f"  Ensemble strategy total return: {final_strat*100:.2f}%")
    print(f"  Buy & hold total return:        {final_bh*100:.2f}%")

    # Optional: plot the equity curves
    plot_backtest(df_backtest)

def plot_backtest(df):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["cum_strategy"], label="Ensemble Strategy", color="blue")
    plt.plot(df.index, df["cum_bh"], label="Buy & Hold", color="gray")
    plt.title("Strategy vs. Buy & Hold")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
