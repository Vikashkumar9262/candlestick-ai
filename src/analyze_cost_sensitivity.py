
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "strategy_research_trades.csv"
REPORTS = ROOT / "reports"

COSTS_BPS = [0, 10, 20, 30]

df = pd.read_csv(INPUT, parse_dates=["signal_date", "exit_date"])

required = {
    "strategy", "threshold", "fold", "signal_date",
    "strategy_gross_return", "market_forward_return",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

rows = []

# Report each fold separately, plus all folds combined.
groups = [
    ("fold", keys, group)
    for keys, group in df.groupby(
        ["strategy", "threshold", "fold"], sort=True
    )
]

groups += [
    ("all_folds", keys, group)
    for keys, group in df.groupby(
        ["strategy", "threshold"], sort=True
    )
]

for scope, keys, group in groups:
    group = group.sort_values("signal_date")

    if scope == "fold":
        strategy, threshold, fold = keys
    else:
        strategy, threshold = keys
        fold = "all"

    for cost_bps in COSTS_BPS:
        cost = cost_bps / 10_000

        strategy_returns = (
            group["strategy_gross_return"].to_numpy(dtype=float) - cost
        )

        # Long-market benchmark over the same selected holding windows.
        benchmark_returns = (
            group["market_forward_return"].to_numpy(dtype=float) - cost
        )

        equity = np.r_[1.0, np.cumprod(1 + strategy_returns)]
        peaks = np.maximum.accumulate(equity)
        drawdowns = equity / peaks - 1

        gains = strategy_returns[strategy_returns > 0].sum()
        losses = -strategy_returns[strategy_returns < 0].sum()

        benchmark_equity = np.prod(1 + benchmark_returns) - 1
        strategy_equity = equity[-1] - 1

        rows.append({
            "scope": scope,
            "strategy": strategy,
            "threshold": threshold,
            "fold": fold,
            "cost_bps": cost_bps,
            "trades": len(strategy_returns),
            "win_rate_pct": (strategy_returns > 0).mean() * 100,
            "mean_trade_net_pct": strategy_returns.mean() * 100,
            "strategy_compounded_pct": strategy_equity * 100,
            "matched_window_benchmark_pct": benchmark_equity * 100,
            "excess_return_pp": (
                strategy_equity - benchmark_equity
            ) * 100,
            "max_drawdown_pct": drawdowns.min() * 100,
            "profit_factor": gains / losses if losses > 0 else np.nan,
        })

result = pd.DataFrame(rows)
result.to_csv(REPORTS / "cost_sensitivity.csv", index=False)

# Print the bullish-only fold results for each assumed cost.
view = result[
    (result["scope"] == "fold")
    & (result["strategy"] == "bullish_only")
]

print("\nBULLISH-ONLY: RESULTS BY FOLD AND COST")
print(view.round(3).to_string(index=False))

print("\nBULLISH-ONLY: COMBINED RESULTS")
overall = result[
    (result["scope"] == "all_folds")
    & (result["strategy"] == "bullish_only")
]
print(overall.round(3).to_string(index=False))

print("\nSaved reports/cost_sensitivity.csv")
print(
    "Benchmark note: this is a matched-window long-market benchmark, "
    "not continuous buy-and-hold."
)
