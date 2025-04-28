"""
ensemble_model_training.py - script to train sub-models and build an ensemble strategy.
"""

import pandas as pd
import numpy as np

from market_analyzer.analyzer import MarketDataAnalyzer
from market_analyzer.experiment_tracker import ExperimentTracker
from market_analyzer.backtester import Backtester

# Our ensemble and sub-strategies:
from market_analyzer.ensemble_strategy import EnsembleStrategy
from market_analyzer.timeseries_nn_strategy import TimeSeriesNNStrategy
from market_analyzer.llm_sentiment_strategy import LLMSentimentStrategy
from market_analyzer.rl_agent_strategy import RLAgentStrategy

def main():
    # 1) Load data
    analyzer = MarketDataAnalyzer()
    analyzer.download_data(period="1y")
    data = analyzer.get_asset_data("BTC-USD")

    # For real usage, data must have columns: ["Open","High","Low","Close","Volume","news_text",...]
    # This is just an example.

    # 2) Initialize sub-strategies
    ts_nn = TimeSeriesNNStrategy(name="time_series_nn", seq_length=30, num_features=5)
    llm_sent = LLMSentimentStrategy(name="llm_sentiment")
    rl_agent = RLAgentStrategy(name="rl_agent")

    # For demonstration, we skip "real" training.
    # e.g.: ts_nn.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)
    #       llm_sent.load_model()
    #       rl_agent.train(data, timesteps=5000)

    # 3) Create an ensemble with majority vote
    ensemble = EnsembleStrategy(
        sub_strategies=[ts_nn, llm_sent, rl_agent],
        voting=True,
        name="ensemble_strategy"
    )

    # 4) Evaluate on test or validation portion
    backtester = Backtester(data, train_size=0.6, test_size=0.2)
    val_results = backtester.evaluate_strategy(ensemble, backtester.validation_data)
    print("[EnsembleStrategy] Validation results:", val_results)

    # 5) Convert series to list to store in experiment tracker
    for key in ["portfolio_value","signals","returns"]:
        if isinstance(val_results.get(key), pd.Series):
            val_results[key] = val_results[key].tolist()

    # 6) Store experiment
    tracker = ExperimentTracker(results_dir="results")
    tracker.save_experiment(
        experiment_name="advanced_ensemble",
        model_name="EnsembleStrategy",
        train_metrics={},
        test_metrics={},
        validation_metrics=val_results,
        params={
            "sub_strategies": ["time_series_nn","llm_sentiment","rl_agent"],
            "voting": True
        }
    )

if __name__=="__main__":
    main()
