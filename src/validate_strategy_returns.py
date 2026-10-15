from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "nifty50.csv"
TRADES_PATH = ROOT / "reports" / "strategy_research_trades.csv"

HORIZON = 5
TOLERANCE = 1e-8

raw = pd.read_csv(RAW_PATH)
trades = pd.read_csv(TRADES_PATH)

# Find the date column regardless of capitalization.
date_col = next(
    (c for c in raw.columns if c.lower() in {"date", "datetime"}),
    None,
)
if date_col is None:
    raise ValueError(f"No date column found. Columns: {raw.columns.tolist()}")

raw["_date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()
raw = (
    raw.dropna(subset=["_date"])
    .sort_values("_date")
    .drop_duplicates("_date")
    .reset_index(drop=True)
)

trades["signal_date"] = pd.to_datetime(
    trades["signal_date"], errors="coerce"
).dt.normalize()
trades["exit_date"] = pd.to_datetime(
    trades["exit_date"], errors="coerce"
).dt.normalize()

date_to_index = {
    date: index for index, date in enumerate(raw["_date"])
}

checks = []

for trade in trades.itertuples(index=False):
    signal_index = date_to_index.get(trade.signal_date)

    if signal_index is None:
        checks.append({
            "signal_date": trade.signal_date,
            "problem": "Signal date missing from raw data",
        })
        continue

    entry_index = signal_index + 1
    exit_index = signal_index + HORIZON

    if exit_index >= len(raw):
        checks.append({
            "signal_date": trade.signal_date,
            "problem": "Exit index outside raw data",
        })
        continue

    entry_open = float(raw.loc[entry_index, "Open"])
    exit_close = float(raw.loc[exit_index, "Close"])
    expected_exit_date = raw.loc[exit_index, "_date"]

    expected_return = exit_close / entry_open - 1.0
    recorded_return = float(trade.market_forward_return)

    return_matches = np.isclose(
        expected_return, recorded_return, atol=TOLERANCE, rtol=0
    )
    date_matches = trade.exit_date == expected_exit_date

    if not return_matches or not date_matches:
        checks.append({
            "signal_date": trade.signal_date,
            "recorded_exit_date": trade.exit_date,
            "expected_exit_date": expected_exit_date,
            "recorded_return": recorded_return,
            "expected_return": expected_return,
            "return_matches": return_matches,
            "date_matches": date_matches,
            "problem": "Return or exit-date mismatch",
        })

print(f"Trades checked: {len(trades)}")
print(f"Mismatches: {len(checks)}")

if checks:
    print(pd.DataFrame(checks).head(20).to_string(index=False))
else:
    print("PASS: all trade returns and exit dates match raw OHLC data.")