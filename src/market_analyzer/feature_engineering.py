"""
Advanced feature engineering for market data.
"""

import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self):
        self.price_windows = [5, 10, 20, 50, 100]
        self.volatility_windows = [5, 10, 20]
        self.momentum_windows = [5, 10, 20]

    def process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method that orchestrates all feature creation.
        Returns a DataFrame of features with the same index as 'data'.
        """
        all_features = pd.DataFrame(index=data.index)

        # Combine separate sets of features
        basic = self.create_basic_features(data)
        tech = self.create_technical_features(data)
        vol  = self.create_volatility_features(data)
        # etc.

        for df_feat in [basic, tech, vol]:
            all_features = all_features.join(df_feat, how="outer")

        # Final cleanup
        all_features.fillna(method='ffill', inplace=True)
        all_features.fillna(method='bfill', inplace=True)
        all_features.fillna(0, inplace=True)

        return all_features

    def create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=data.index)
        feats['returns'] = data['Close'].pct_change()
        feats['log_returns'] = np.log1p(data['Close']).diff()
        feats['volume_change'] = data['Volume'].pct_change()
        # ...
        return feats

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=data.index)

        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(data['Close'])
        feats['rsi'] = rsi_indicator.rsi()

        # MACD
        macd = ta.trend.MACD(data['Close'])
        feats['macd'] = macd.macd()
        feats['macd_signal'] = macd.macd_signal()
        feats['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['Close'])
        feats['bb_high'] = bb.bollinger_hband()
        feats['bb_low']  = bb.bollinger_lband()
        feats['bb_width'] = bb.bollinger_wband()

        return feats

    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=data.index)
        returns = data['Close'].pct_change()

        for window in self.volatility_windows:
            feats[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
            # ...
        return feats
