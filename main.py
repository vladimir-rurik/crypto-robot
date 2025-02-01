# main.py

import time
from coingecko_client import get_intraday_ohlcv
from onchain_client import get_onchain_metrics
from db_utils import get_db_connection, write_dataframe_to_sql, write_onchain_to_sql
from config import DB_CONFIG  # Import the database configuration

if __name__ == "__main__":
    # Map CoinGecko IDs to short names (symbols)
    crypto_ids = {
        'ethereum': 'ETH',
        'solana': 'SOL',
        'cardano': 'ADA',
        'chainlink': 'LINK',
        'avalanche-2': 'AVAX',
        'stellar': 'XLM',
        'litecoin': 'LTC',
        'polkadot': 'DOT',
        'uniswap': 'UNI',
        'aave': 'AAVE',
        'the-sandbox': 'SAND',
        'axie-infinity': 'AXS',
        'matic-network': 'MATIC',
        'fantom': 'FTM'
    }
    
    # Use database configuration from config.py
    conn = get_db_connection(
        DB_CONFIG['SERVER'],
        DB_CONFIG['DATABASE'],
        DB_CONFIG['USERNAME'],
        DB_CONFIG['PASSWORD']
    )
    
    for coin_id, symbol in crypto_ids.items():
        print(f"Processing {coin_id} ({symbol})...")
        
        try:
            # 4. Retrieve intraday OHLCV data
            ohlcv_df = get_intraday_ohlcv(coin_id, vs_currency='eur', days=365)
            if ohlcv_df.empty:
                print(f"⚠️ No data available for {coin_id}. Skipping...")
                continue
            
            # 5. Write OHLCV data to SQL (table: intraday_data)
            # Make sure your table columns match: (timestamp, open_price, high_price, low_price, close_price, volume)
            write_dataframe_to_sql(ohlcv_df, 'intraday_data', conn, symbol)

            # 6. Retrieve on-chain metrics (placeholder)
            onchain_df = get_onchain_metrics(symbol)
            
            # 7. Write on-chain metrics to SQL (table: onchain_data)
            write_onchain_to_sql(onchain_df, 'onchain_data', conn, symbol)

        except Exception as e:
            print(f"❌ Error processing {coin_id}: {e}")
        
        # Respect rate limits
        time.sleep(2)
    
    conn.close()
    print("Data collection complete!")
