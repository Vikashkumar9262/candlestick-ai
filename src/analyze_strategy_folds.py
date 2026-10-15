
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "reports" / "strategy_research_trades.csv"

df = pd.read_csv(path, parse_dates=["signal_date"])
if df.empty:
    raise ValueError("No trades found in strategy_research_trades.csv")

results = []

for (strategy, threshold, fold), group in df.groupby(
    ["strategy", "threshold", "fold"], dropna=False
):
    group = group.sort_values("signal_date")
    returns = group["strategy_net_return"].to_numpy(dtype=float)

    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    results.append({
        "strategy": strategy,
        "threshold": threshold,
        "fold": fold,
        "trades": len(returns),
        "win_rate_pct": (returns > 0).mean() * 100,
        "mean_net_return_pct": returns.mean() * 100,
        "compounded_return_pct": (equity[-1] - 1) * 100,
        "max_drawdown_pct": drawdown.min() * 100,
        "profit_factor": gains / losses if losses > 0 else np.nan,
    })

out = pd.DataFrame(results).sort_values(
    ["strategy", "threshold", "fold"]
)

out.to_csv(ROOT / "reports" / "strategy_fold_metrics.csv", index=False)

print(
    out[out["strategy"] == "bullish_only"]
    .round(3)
    .to_string(index=False)
)
print("\nSaved reports/strategy_fold_metrics.csv")
