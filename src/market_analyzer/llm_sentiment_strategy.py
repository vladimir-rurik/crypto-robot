"""
llm_sentiment_strategy.py
Uses a Hugging Face Transformers sentiment model to produce signals.
"""

import pandas as pd
import numpy as np
from typing import Optional
from transformers import pipeline
from .strategy import TradingStrategy

class LLMSentimentStrategy(TradingStrategy):
    """
    A 'real' LLM-based sentiment strategy:
     - load a huggingface sentiment classifier
     - data must have 'news_text' column
     - negative => -1, neutral => 0, positive => +1
    """

    def __init__(self, name="llm_sentiment", hf_model="distilbert-base-uncased-finetuned-sst-2-english"):
        super().__init__(name)
        self.hf_model = hf_model
        self.classifier = None

    def load_model(self):
        """Actually load the huggingface pipeline once."""
        self.classifier = pipeline("sentiment-analysis", model=self.hf_model)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=data.index)
        if self.classifier is None:
            print("[LLMSentimentStrategy] No classifier loaded, returning zeros.")
            return signals

        if "news_text" not in data.columns:
            print("[LLMSentimentStrategy] 'news_text' column missing, returning zeros.")
            return signals

        for idx in data.index:
            text = data.at[idx, "news_text"]
            if not text or not isinstance(text, str):
                continue
            # e.g. result=[{"label": "POSITIVE", "score": 0.999}]
            result = self.classifier(text[:512])[0]
            label = result["label"].upper()
            if label=="NEGATIVE":
                signals.loc[idx] = -1
            elif label=="POSITIVE":
                signals.loc[idx] = +1
            else:
                signals.loc[idx] = 0

        return signals
