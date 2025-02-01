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

def write_dataframe_to_sql(df: pd.DataFrame, table_name: str, connection):
    """
    Inserts a pandas DataFrame into a specified SQL table.
    Expects that the table already exists with appropriate columns.
    """
    cursor = connection.cursor()
    
    # Example insertion, row by row. 
    # For large data, consider bulk insert methods or SQLAlchemy's to_sql with fast_executemany.
    
    for index, row in df.iterrows():
        # Assuming columns match exactly.
        # Example for 'intraday_data' table: (timestamp, open, high, low, close, volume)
        insert_query = f"""
            INSERT INTO {table_name} (timestamp, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, 
                       row['timestamp'], 
                       row['open'], 
                       row['high'], 
                       row['low'], 
                       row['close'],
                       row['volume'])
    connection.commit()

def write_onchain_to_sql(df: pd.DataFrame, table_name: str, connection):
    """
    Similar approach for on-chain metrics.
    """
    cursor = connection.cursor()
    
    for index, row in df.iterrows():
        insert_query = f"""
            INSERT INTO {table_name} (timestamp, transaction_count, active_wallets)
            VALUES (?, ?, ?)
        """
        cursor.execute(insert_query, 
                       row['timestamp'], 
                       row['transaction_count'], 
                       row['active_wallets'])
    connection.commit()
