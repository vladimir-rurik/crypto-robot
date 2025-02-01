# db_utils.py

import pyodbc
import pandas as pd

def get_db_connection(server, database, username, password, driver='{ODBC Driver 17 for SQL Server}'):
    """
    Create a connection to MS SQL Server using pyodbc.
    """
    conn_str = (
        f'DRIVER={driver};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password}'
    )
    connection = pyodbc.connect(conn_str)
    return connection

def write_dataframe_to_sql(df: pd.DataFrame, table_name: str, connection, symbol: str):
    """
    Inserts a pandas DataFrame into a specified SQL table.
    Expects that the table already exists with columns including a symbol.
    """
    cursor = connection.cursor()
    # insertion, row by row. 
    # For large data, consider bulk insert methods or SQLAlchemy's to_sql with fast_executemany.    
    for index, row in df.iterrows():
        # Assuming columns match exactly.
        insert_query = f"""
            INSERT INTO {table_name} (timestamp, symbol, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, 
                       row['timestamp'], 
                       symbol,
                       row['open'], 
                       row['high'], 
                       row['low'], 
                       row['close'],
                       row['volume'])
    connection.commit()

def write_onchain_to_sql(df: pd.DataFrame, table_name: str, connection, symbol: str):
    """
    Inserts on-chain data into SQL including the symbol.
    """
    cursor = connection.cursor()
    
    for index, row in df.iterrows():
        insert_query = f"""
            INSERT INTO {table_name} (timestamp, symbol, transaction_count, active_wallets)
            VALUES (?, ?, ?, ?)
        """
        cursor.execute(insert_query, 
                       row['timestamp'], 
                       symbol,
                       row['transaction_count'], 
                       row['active_wallets'])
    connection.commit()

