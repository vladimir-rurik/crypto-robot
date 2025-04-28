"""
ensemble_dashboard.py
Compare the final ensemble strategy vs. sub-models in a simple dashboard.
"""

import pandas as pd
import numpy as np
from market_analyzer.dashboard import StrategyDashboard

def main():
    # Here we mock some data. In real usage, gather actual metrics from the backtester.
    import datetime
    dates = pd.date_range(datetime.date(2025,1,1), periods=50, freq="D")

    def random_portfolio(dates):
        returns = np.random.normal(0, 0.001, len(dates))
        cum = (1 + returns).cumprod() * 10000
        return pd.Series(cum, index=dates)

    # Suppose we have 4 strategies: time_series_nn, llm_sentiment, rl_agent, ensemble_strategy
    strategies = ["time_series_nn","llm_sentiment","rl_agent","ensemble_strategy"]
    results = []
    for strat in strategies:
        pv = random_portfolio(dates)
        ret_s = pv.pct_change().fillna(0)
        total_ret = float(pv.iloc[-1]/pv.iloc[0] - 1)
        results.append({
            "strategy_name": strat,
            "portfolio_value": pv,
            "returns": ret_s,
            "total_return": total_ret,
            "annual_return": 0.1,
            "volatility": 0.02,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15
        })

    # Now pass them to StrategyDashboard
    dash = StrategyDashboard(figsize=(15,10))
    dash.plot_summary(results)

if __name__=="__main__":
    main()
