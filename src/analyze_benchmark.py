
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "nifty50.csv"
PRED_PATH = ROOT / "reports" / "walk_forward_predictions.csv"
TRADES_PATH = ROOT / "reports" / "strategy_research_trades.csv"

HORIZON = 5
COSTS_BPS = [0, 10, 20, 30]

raw = pd.read_csv(RAW_PATH)
pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_date"])

date_col = next(
    c for c in raw.columns
    if c.lower() in {"date", "datetime"}
)
raw["_date"] = pd.to_datetime(raw[date_col]).dt.normalize()
raw = (
    raw.sort_values("_date")
    .drop_duplicates("_date")
    .reset_index(drop=True)
)
pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
trades["signal_date"] = pd.to_datetime(
    trades["signal_date"]
).dt.normalize()

date_to_idx = {d: i for i, d in enumerate(raw["_date"])}

# Build one continuous long-market return per complete test fold.
benchmarks = {}
for fold, group in pred.groupby("fold"):
    dates = group["date"].dropna().sort_values()
    if dates.empty:
        continue

    first_signal_idx = date_to_idx[dates.iloc[0]]
    last_signal_idx = date_to_idx[dates.iloc[-1]]

    entry_idx = first_signal_idx + 1
    exit_idx = last_signal_idx + HORIZON

    if exit_idx >= len(raw):
        raise ValueError(f"Benchmark exit outside raw data for fold {fold}")

    entry = float(raw.loc[entry_idx, "Open"])
    exit_price = float(raw.loc[exit_idx, "Close"])

    benchmarks[fold] = {
        "start_date": raw.loc[entry_idx, "_date"],
        "end_date": raw.loc[exit_idx, "_date"],
        "gross_return": exit_price / entry - 1.0,
    }

results = []

for (strategy, threshold), all_trades in trades.groupby(
    ["strategy", "threshold"]
):
    for cost_bps in COSTS_BPS:
        cost = cost_bps / 10_000
        fold_strategy_returns = []
        fold_benchmark_returns = []

        for fold, benchmark in benchmarks.items():
            group = all_trades[all_trades["fold"] == fold].sort_values(
                "signal_date"
            )

            # A fold with no trades stays in cash: return is zero.
            net = (
                group["strategy_gross_return"].to_numpy(dtype=float)
                - cost
            )
            strategy_return = (
                np.prod(1.0 + net) - 1.0 if len(net) else 0.0
            )

            # One entry and one exit for continuous long exposure.
            benchmark_return = benchmark["gross_return"] - cost

            fold_strategy_returns.append(strategy_return)
            fold_benchmark_returns.append(benchmark_return)

            results.append({
                "scope": "fold",
                "strategy": strategy,
                "threshold": threshold,
                "fold": fold,
                "cost_bps": cost_bps,
                "trades": len(group),
                "strategy_return_pct": strategy_return * 100,
                "continuous_long_benchmark_pct": benchmark_return * 100,
                "excess_return_pp": (
                    strategy_return - benchmark_return
                ) * 100,
                "benchmark_start": benchmark["start_date"],
                "benchmark_end": benchmark["end_date"],
            })

        strategy_total = np.prod(
            1.0 + np.array(fold_strategy_returns)
        ) - 1.0
        benchmark_total = np.prod(
            1.0 + np.array(fold_benchmark_returns)
        ) - 1.0

        results.append({
            "scope": "all_folds",
            "strategy": strategy,
            "threshold": threshold,
            "fold": "all",
            "cost_bps": cost_bps,
            "trades": len(all_trades),
            "strategy_return_pct": strategy_total * 100,
            "continuous_long_benchmark_pct": benchmark_total * 100,
            "excess_return_pp": (
                strategy_total - benchmark_total
            ) * 100,
            "benchmark_start": None,
            "benchmark_end": None,
        })

out = pd.DataFrame(results)
out.to_csv(ROOT / "reports" / "benchmark_comparison.csv", index=False)

print("\nBULLISH-ONLY: COMBINED COMPARISON")
print(
    out[
        (out["scope"] == "all_folds")
        & (out["strategy"] == "bullish_only")
    ].round(3).to_string(index=False)
)

print("\nBULLISH-ONLY: FOLD COMPARISON AT 10 BPS")
print(
    out[
        (out["scope"] == "fold")
        & (out["strategy"] == "bullish_only")
        & (out["cost_bps"] == 10)
    ].round(3).to_string(index=False)
)

print("\nSaved reports/benchmark_comparison.csv")
print(
    "Note: benchmark is continuous long exposure across each fold's "
    "test window. Returns exclude dividends and financing costs."
)
