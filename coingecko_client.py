# coingecko_client.py

import time
from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime

cg = CoinGeckoAPI()

def get_intraday_ohlcv(coin_id: str, vs_currency: str = 'usd', days: int = 1) -> pd.DataFrame:
    """
    Fetch intraday market data (prices, volumes) from CoinGecko.
    
    :param coin_id: CoinGecko coin id, e.g., 'ethereum', 'solana' etc.
    :param vs_currency: currency to compare with, e.g. 'usd'
    :param days: how many days back to retrieve with intraday granularity
    :return: DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    # Using 'cg.get_coin_market_chart_by_id' to get ohlc data for up to 'days' days 
    # with an ~hourly granularity (CoinGecko typically returns up to 24 data points for 1 day).
    
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        # data is a dict with keys: 'prices', 'market_caps', 'total_volumes'
        # each is a list of [timestamp, value]
    except ValueError as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    prices = data['prices']          # timestamp in ms, price
    volumes = data['total_volumes']  # timestamp in ms, volume
    
    # Convert to DataFrame
    df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
    
    # Merge on timestamp
    df = pd.merge(df_prices, df_volumes, on='timestamp', how='inner')
    
    # Convert timestamp from ms to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns={'price': 'close'}, inplace=True)
    
    # WARNING: CoinGecko does not provide direct "open", "high", "low" in this endpoint for intraday. 
    # This is a best-effort approach to transform the close price data. 
    # Often you'd want the /coins/{id}/ohlc endpoint for real OHLC, but that is daily or limited. 
    # So we approximate or rely on the single price point as "close".
    df['open'] = df['close']
    df['high'] = df['close']
    df['low'] = df['close']
    
    # Reorder columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

