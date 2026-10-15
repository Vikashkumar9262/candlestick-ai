
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}


def download_market_data(
    ticker: str,
    start: str = "2010-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    # yfinance can return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    columns = required + (
        ["Volume"] if "Volume" in df.columns else []
    )
    df = df[columns].copy()

    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=required)

    df = df[
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
    ]

    return df


if __name__ == "__main__":
    for name, ticker in TICKERS.items():
        df = download_market_data(ticker)
        path = DATA_DIR / f"{name.lower()}.csv"
        df.to_csv(path, index_label="Date")
        print(f"{name}: {len(df)} rows saved to {path}")
