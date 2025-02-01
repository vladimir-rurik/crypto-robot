# main.py

import time
import pandas as pd
from coingecko_client import get_intraday_ohlcv
from onchain_client import get_onchain_metrics
from db_utils import get_db_connection, write_dataframe_to_sql, write_onchain_to_sql

if __name__ == "__main__":
    # 1. List of the 14 cryptocurrencies (by CoinGecko ID)
    crypto_ids = {
        'ethereum': 'ETH',
        'solana': 'SOL',
        'cardano': 'ADA',
        'chainlink': 'LINK',
        'avalanche': 'AVAX',
        'stellar': 'XLM',
        'litecoin': 'LTC',
        'polkadot': 'DOT',
        'uniswap': 'UNI',
        'aave': 'AAVE',
        'the-sandbox': 'SAND',
        'axie-infinity': 'AXS',
        'polygon': 'MATIC',
        'fantom': 'FTM'
    }
    
    # 2. Setup database connection details
    SERVER   = 'localhost'
    DATABASE = 'crypto_robot'
    USERNAME = 'sa'
    PASSWORD = '1StrongPwdexit'
    
    # 3. Connect to MSSQL
    conn = get_db_connection(SERVER, DATABASE, USERNAME, PASSWORD)
    
    for coin_id, symbol in crypto_ids.items():
        print(f"Processing {coin_id}...")
        
        # 4. Retrieve intraday OHLCV data
        ohlcv_df = get_intraday_ohlcv(coin_id, vs_currency='usd', days=1)
        
        # 5. Write OHLCV data to SQL (table: intraday_data)
        # Make sure your table columns match: (timestamp, open_price, high_price, low_price, close_price, volume)
        write_dataframe_to_sql(ohlcv_df, 'intraday_data', conn)
        
        # 6. Retrieve on-chain metrics (placeholder)
        onchain_df = get_onchain_metrics(symbol)
        
        # 7. Write on-chain metrics to SQL (table: onchain_data)
        write_onchain_to_sql(onchain_df, 'onchain_data', conn)
        
        # OPTIONAL: Sleep or manage rate limit 
        time.sleep(2)
    
    conn.close()
    print("Data collection complete!")
