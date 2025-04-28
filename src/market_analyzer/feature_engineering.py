"""
Advanced feature engineering for market data.
"""

import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    """Feature engineering logic: basic, technical, volatility, etc."""

    def __init__(self):
        self.price_windows = [5, 10, 20, 50, 100]
        self.volatility_windows = [5, 10, 20]
        self.momentum_windows = [5, 10, 20]

    def process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to orchestrate feature creation.
        Returns a DataFrame with new columns.
        """
        all_features = pd.DataFrame(index=data.index)

        # Combine various sets of features
        basic_features = self.create_basic_features(data)
        tech_features  = self.create_technical_features(data)
        vol_features   = self.create_volatility_features(data)
        
        for feat_df in [basic_features, tech_features, vol_features]:
            if not feat_df.empty:
                all_features = all_features.join(feat_df, how='outer')

        # Fill any final NaNs
        all_features.fillna(method='ffill', inplace=True)
        all_features.fillna(method='bfill', inplace=True)
        all_features.fillna(0, inplace=True)

        return all_features

    def create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=data.index)
        feats['returns'] = data['Close'].pct_change()
        feats['log_returns'] = np.log1p(data['Close']).diff()
        # ...
        return feats

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=data.index)

        rsi = ta.momentum.RSIIndicator(data['Close'], fillna=True)
        feats['rsi'] = rsi.rsi()

        macd = ta.trend.MACD(data['Close'], fillna=True)
        feats['macd'] = macd.macd()
        feats['macd_signal'] = macd.macd_signal()
        feats['macd_diff'] = macd.macd_diff()

        # Bollinger
        bb = ta.volatility.BollingerBands(data['Close'], fillna=True)
        feats['bb_high'] = bb.bollinger_hband()
        feats['bb_low']  = bb.bollinger_lband()
        feats['bb_width'] = bb.bollinger_wband()

        return feats

    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=data.index)
        returns = data['Close'].pct_change()

        for window in self.volatility_windows:
            feats[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)

        return feats
