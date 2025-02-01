
IF EXISTS (SELECT * FROM sys.tables WHERE name = 'intraday_data')
    DROP TABLE intraday_data;
CREATE TABLE intraday_data (
    id INT IDENTITY PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(5) NOT NULL,  -- column for cryptocurrency short name
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume FLOAT
);


IF EXISTS (SELECT * FROM sys.tables WHERE name = 'onchain_data')
    DROP TABLE onchain_data;
CREATE TABLE onchain_data (
    id INT IDENTITY PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(5) NOT NULL,  -- column for cryptocurrency short name
    transaction_count BIGINT,
    active_wallets BIGINT
);
