
from pathlib import Path

import numpy as np
import pandas as pd

REPORT_DIR = Path("reports")
INPUT_FILE = REPORT_DIR / "test_predictions.csv"

HORIZON = 5

# Illustrative round-trip costs, in basis points.
COST_SCENARIOS_BPS = [5, 10, 20]

# Positive = long; negative = short; zero = no trade.
DIRECTIONS = {
    "bullish_engulfing": 1,
    "hammer": 1,
    "bearish_engulfing": -1,
    "shooting_star": -1,
}

OUTPUT_TRADES = REPORT_DIR / "backtest_trades.csv"
OUTPUT_SUMMARY = REPORT_DIR / "backtest_summary.csv"


def compound_return(returns):
    """Compound a sequence of simple trade returns."""
    returns = np.asarray(returns, dtype=float)

    if len(returns) == 0:
        return np.nan

    return float(np.prod(1.0 + returns) - 1.0)


def max_drawdown(returns):
    """Maximum drawdown on the sequential trade-equity curve."""
    returns = np.asarray(returns, dtype=float)

    if len(returns) == 0:
        return np.nan

    equity = np.concatenate([
        [1.0],
        np.cumprod(1.0 + returns),
    ])
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0

    return float(drawdowns.min())


def profit_factor(returns):
    returns = np.asarray(returns, dtype=float)
    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return float(gross_profit / gross_loss)


def summarize(returns):
    """Trade-level metrics; Sharpe is deliberately not annualized."""
    returns = np.asarray(returns, dtype=float)

    if len(returns) == 0:
        return {
            "trades": 0,
            "total_return": np.nan,
            "mean_trade_return": np.nan,
            "median_trade_return": np.nan,
            "win_rate": np.nan,
            "trade_return_std": np.nan,
            "trade_level_sharpe": np.nan,
            "max_drawdown": np.nan,
            "profit_factor": np.nan,
        }

    std = returns.std(ddof=1) if len(returns) > 1 else np.nan

    return {
        "trades": len(returns),
        "total_return": compound_return(returns),
        "mean_trade_return": float(returns.mean()),
        "median_trade_return": float(np.median(returns)),
        "win_rate": float((returns > 0).mean()),
        "trade_return_std": float(std),
        "trade_level_sharpe": (
            float(returns.mean() / std)
            if np.isfinite(std) and std > 0
            else np.nan
        ),
        "max_drawdown": max_drawdown(returns),
        "profit_factor": profit_factor(returns),
    }


def select_non_overlapping_trades(df):
    """
    Select pattern signals in chronological order.

    A signal at row i uses the existing forward return for that row.
    A new signal may be accepted at row i + HORIZON, after the previous
    five-candle holding window ends.

    This assumes rows are in chronological order and each row represents
    one consecutive candle.
    """
    directions = (
        df["predicted_pattern"]
        .astype(str)
        .map(DIRECTIONS)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    accepted = []
    next_allowed = 0

    for i, direction in enumerate(directions):
        if direction == 0 or i < next_allowed:
            continue

        accepted.append(i)
        next_allowed = i + HORIZON

    return accepted


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run `python -m src.evaluate` first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"predicted_pattern", "forward_return"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {INPUT_FILE}: {sorted(missing)}"
        )

    # Do not sort unless a reliable timestamp column is present and validated.
    # The evaluator should already write predictions in chronological order.
    df["forward_return"] = pd.to_numeric(
        df["forward_return"], errors="coerce"
    )
    df = df.dropna(
        subset=["predicted_pattern", "forward_return"]
    ).reset_index(drop=True)

    if not np.isfinite(df["forward_return"].to_numpy()).all():
        raise ValueError("Non-finite forward returns found.")

    if (df["forward_return"] <= -1).any():
        raise ValueError(
            "A simple return <= -100% was found; check the return calculation."
        )

    # Check that the rows represent consecutive candles. If there are
    # gaps in the test set, the cooldown logic needs to be adapted.
    selected_indices = select_non_overlapping_trades(df)

    if not selected_indices:
        raise ValueError(
            "No eligible pattern trades were found in the test set."
        )

    trades = df.iloc[selected_indices].copy()
    trades["test_row"] = selected_indices
    trades["direction"] = (
        trades["predicted_pattern"]
        .astype(str)
        .map(DIRECTIONS)
        .astype(int)
    )

    # Existing forward_return is the underlying long return over the
    # five-candle window. Short return is approximated as its negative.
    # This is a first-pass proxy, not exact futures P&L.
    trades["underlying_forward_return"] = trades["forward_return"]
    trades["gross_strategy_return"] = (
        trades["direction"] * trades["underlying_forward_return"]
    )
    trades["gross_long_benchmark_return"] = (
        trades["underlying_forward_return"]
    )

    all_summaries = []
    all_trade_versions = []

    for cost_bps in COST_SCENARIOS_BPS:
        cost = cost_bps / 10_000.0

        version = trades.copy()
        version["cost_bps_round_trip"] = cost_bps
        version["strategy_net_return"] = (
            version["gross_strategy_return"] - cost
        )
        version["long_benchmark_net_return"] = (
            version["gross_long_benchmark_return"] - cost
        )

        strategy_metrics = summarize(version["strategy_net_return"])
        benchmark_metrics = summarize(
            version["long_benchmark_net_return"]
        )

        all_summaries.append({
            "cost_bps_round_trip": cost_bps,
            "strategy": "pattern_long_short",
            **strategy_metrics,
        })
        all_summaries.append({
            "cost_bps_round_trip": cost_bps,
            "strategy": "long_same_trade_windows",
            **benchmark_metrics,
        })

        all_trade_versions.append(version)

    summary = pd.DataFrame(all_summaries)
    trade_report = pd.concat(all_trade_versions, ignore_index=True)

    summary.to_csv(OUTPUT_SUMMARY, index=False)
    trade_report.to_csv(OUTPUT_TRADES, index=False)

    print("\n=== Direction-aware backtest ===")
    print(
        summary.round(6).to_string(index=False)
    )

    print("\nPattern signals accepted:", len(selected_indices))
    print("Rows in held-out test set:", len(df))
    print("\nSaved reports:")
    print(f"- {OUTPUT_SUMMARY}")
    print(f"- {OUTPUT_TRADES}")


if __name__ == "__main__":
    main()
