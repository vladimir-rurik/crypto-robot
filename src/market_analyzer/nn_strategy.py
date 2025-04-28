"""
nn_strategy.py
A simplified Keras-based strategy for classification in {-1,0,1}.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from .strategy import TradingStrategy
from .utils import validate_data  # if you have a validate_data() function

class NeuralNetworkStrategy(TradingStrategy):
    """
    A neural network-based trading strategy that outputs signals in {-1,0,1}.
    """

    def __init__(self, params: Dict = None):
        super().__init__("nn_strategy")
        default_params = {
            "seq_length": 30,
            "num_features": 5,
            "n_hidden": 64,
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 1e-3
        }
        self.params = {**default_params, **(params or {})}
        self.model = None

    def build_model(self):
        """
        Build a simple LSTM-based Keras model for classification.
        """
        seq_len  = self.params["seq_length"]
        num_feat = self.params["num_features"]
        n_hidden = self.params["n_hidden"]

        model = Sequential()
        model.add(LSTM(n_hidden, input_shape=(seq_len, num_feat)))
        model.add(Dense(3, activation="softmax"))  # 3 classes => Sell,Hold,Buy
        opt = Adam(learning_rate=self.params["learning_rate"])
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the neural network. Return a dict with final metrics."""
        if self.model is None:
            self.build_model()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            verbose=1
        )
        final_val_acc = history.history["val_accuracy"][-1]
        return {"final_val_acc": float(final_val_acc)}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals in {-1, 0, 1}."""
        try:
            data = validate_data(data)  # if you have a function for data checks
        except ValueError as e:
            print("[NN Strategy] Validation error:", e)
            return pd.Series(0.0, index=getattr(data, "index", pd.RangeIndex(0)))

        if data.empty:
            return pd.Series(0.0, index=data.index)

        if self.model is None:
            print("[NN Strategy] Model not trained; returning zeros.")
            return pd.Series(0.0, index=data.index)

        X = self._prepare_inference_data(data)
        if X is None or len(X)==0:
            return pd.Series(0.0, index=data.index)

        preds = self.model.predict(X)
        class_idx = preds.argmax(axis=1)  # {0,1,2}
        # Map 0->-1, 1->0, 2->1
        mapping = {0:-1, 1:0, 2:1}
        mapped = [mapping[i] for i in class_idx]

        signals = pd.Series(0.0, index=data.index)
        offset = self.params["seq_length"] - 1
        pred_index = data.index[offset : offset + len(mapped)]
        signals.loc[pred_index] = mapped
        return signals

    def _prepare_inference_data(self, data: pd.DataFrame):
        seq_len = self.params["seq_length"]
        feat_cols = ["Close","High","Low","Open","Volume"]
        if any(c not in data.columns for c in feat_cols):
            print("[NN Strategy] Missing columns for inference.")
            return None

        arr = data[feat_cols].values
        if len(arr) < seq_len:
            return None

        windows = []
        for i in range(seq_len, len(arr)+1):
            window = arr[i-seq_len:i]
            windows.append(window)
        return np.array(windows, dtype=np.float32)
