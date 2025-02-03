"""
rl_agent_strategy.py
Uses stable-baselines or a custom RL approach to produce signals in {-1,0,1}.
"""

import pandas as pd
import numpy as np
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from .strategy import TradingStrategy

class DummyTradingEnv:
    """
    Minimal environment for demonstration. Real code would define an obs space,
    action space, rewards, etc.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return np.zeros(5, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = (self.current_step >= len(self.data))
        obs = np.zeros(5, dtype=np.float32)
        reward = 0.0
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

class RLAgentStrategy(TradingStrategy):
    """
    A discrete-action RL-based strategy using stable-baselines PPO.
    Actions => 0=Sell,1=Hold,2=Buy => signals => -1,0,+1
    """
    def __init__(self, name="rl_agent"):
        super().__init__(name)
        self.model = None

    def train(self, data: pd.DataFrame, timesteps=10000):
        env = DummyTradingEnv(data)
        def make_env():
            return env
        vec_env = DummyVecEnv([make_env])

        self.model = PPO("MlpPolicy", vec_env, verbose=0)
        self.model.learn(total_timesteps=timesteps)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=data.index)
        if self.model is None:
            print("[RLAgent] Model not trained, returning zeros.")
            return signals

        # We'll do a simplistic approach: step through each row, get an action
        obs = np.zeros((1,5), dtype=np.float32)
        for i, idx in enumerate(data.index):
            action, _ = self.model.predict(obs)
            mapping = {0:-1, 1:0, 2:1}
            signals.loc[idx] = mapping[action]
            # Next obs => omitted for brevity
        return signals
