import os, requests, datetime as dt, pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("BSCSCAN_KEY")
SQL_URI = os.getenv("MSSQL_URI")          # mssql+pyodbc://…

SYMBOL = "BNB"
start_date = dt.date(2024, 1, 1)
end_date   = dt.date.today()

BASE_URL = "https://api.bscscan.com/api"  # <-- correct domain, no v2 path

def fetch(action: str, field: str) -> pd.DataFrame:
    params = {
        "module":    "stats",
        "action":    action,               # e.g. dailytransactioncount
        "startdate": start_date.isoformat(),
        "enddate":   end_date.isoformat(),
        "sort":      "asc",
        "apikey":    API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()                  # HTTP != 200 → exception
    payload = r.json()

    if payload.get("status") != "1":
        raise RuntimeError(
            f"BscScan returned error: {payload.get('message')} – {payload.get('result')}"
        )

    df = pd.DataFrame(payload["result"])[["UTCDate", field]]
    df.rename(columns={"UTCDate": "timestamp", field: action}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[action] = df[action].astype("int64")
    return df

# --- the two downloads -----------------------------------------
tx_df   = fetch("dailytx",          "transactionCount")  # was dailytransactioncount
addr_df = fetch("dailynewaddress",  "newAddress")        # was dailynewaddress + field


df = (
    tx_df.merge(addr_df, on="timestamp")
         .rename(columns={
             "dailytransactioncount": "transaction_count",
             "dailynewaddress":       "active_wallets"
         })
)
df["symbol"] = SYMBOL
df = df[["timestamp", "symbol", "transaction_count", "active_wallets"]]

# ---- push to SQL Server -----------------------------------------------------
engine = create_engine(SQL_URI, fast_executemany=True)

with engine.begin() as conn:
    existing = set(
        d[0] for d in conn.execute(
            text("SELECT timestamp FROM crypto_robot.dbo.onchain_data WHERE symbol=:s"),
            {"s": SYMBOL}
        )
    )
    to_insert = df[~df["timestamp"].isin(existing)]

    to_insert.to_sql(
        "onchain_data",
        con=conn,
        schema="dbo",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1_000
    )

print(f"Inserted {len(to_insert)} new rows.")