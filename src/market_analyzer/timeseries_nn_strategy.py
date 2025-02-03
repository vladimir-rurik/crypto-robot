"""
timeseries_nn_strategy.py
A more advanced LSTM-based approach for time-series classification in {sell=0,hold=1,buy=2}.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .strategy import TradingStrategy

class TimeSeriesNNStrategy(TradingStrategy):
    """
    We'll do a classification approach: 3 classes => sell=0, hold=1, buy=2.
    Then generate signals => map 0->-1,1->0,2->+1
    """

    def __init__(self, name="time_series_nn", seq_length=30, num_features=5, n_hidden=64):
        super().__init__(name)
        self.seq_length = seq_length
        self.num_features = num_features
        self.n_hidden = n_hidden
        self.model = None

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs=5,
              batch_size=32,
              lr=1e-3):
        """
        X_* shape = (samples, seq_length, num_features)
        y_* shape = (samples,) in {0,1,2} or (samples,3} if already one-hot
        """
        # Convert y to one-hot if needed
        if y_train.ndim==1:
            y_train_oh = np.zeros((y_train.shape[0], 3), dtype=np.float32)
            for i,cls in enumerate(y_train):
                y_train_oh[i, cls] = 1.0

            y_val_oh = np.zeros((y_val.shape[0], 3), dtype=np.float32)
            for i,cls in enumerate(y_val):
                y_val_oh[i, cls] = 1.0
        else:
            y_train_oh = y_train
            y_val_oh = y_val

        self.model = Sequential()
        self.model.add(LSTM(self.n_hidden, input_shape=(self.seq_length, self.num_features)))
        self.model.add(Dense(3, activation="softmax"))
        opt = Adam(learning_rate=lr)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        es = EarlyStopping(patience=3, restore_best_weights=True)
        hist = self.model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=data.index)
        if self.model is None:
            print("[TimeSeriesNN] No model, returning zeros.")
            return signals

        X = self._build_inference_windows(data)
        if X is None or len(X)==0:
            return signals

        preds = self.model.predict(X)
        classes = preds.argmax(axis=1)
        mapping = {0:-1, 1:0, 2:1}
        mapped = [mapping[c] for c in classes]

        offset = self.seq_length - 1
        n_preds = len(mapped)
        pred_index = data.index[offset:offset+n_preds]
        signals.loc[pred_index] = mapped
        return signals

    def _build_inference_windows(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        feat_cols = ["Close","High","Low","Open","Volume"]
        if any(c not in data.columns for c in feat_cols):
            print("[TimeSeriesNN] Missing required columns.")
            return None

        arr = data[feat_cols].values
        if len(arr) < self.seq_length:
            return None

        out = []
        for i in range(self.seq_length, len(arr)+1):
            window = arr[i-self.seq_length:i]
            out.append(window)
        return np.array(out, dtype=np.float32)
