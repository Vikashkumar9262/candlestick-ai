
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "reports" / "walk_forward_predictions.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "nifty50.csv"
REPORT_DIR = ROOT / "reports"

# Must match the forward-return horizon used by your existing project.
HORIZON = 5

# Illustrative round-trip cost, in basis points.
# 10 bps = 0.10%. Replace with a realistic estimate for your
# actual tradable instrument and execution method.
ROUND_TRIP_COST_BPS = 10.0

# Exploratory thresholds only. Do not use these test results
# to select the final strategy or threshold.
THRESHOLDS = [0.0, 0.70, 0.80, 0.90]

STRATEGIES = {
    "directional": {
        "bullish_engulfing": 1,
        "hammer": 1,
        "bearish_engulfing": -1,
        "shooting_star": -1,
    },
    "bullish_only": {
        "bullish_engulfing": 1,
        "hammer": 1,
    },
    "bearish_only": {
        "bearish_engulfing": -1,
        "shooting_star": -1,
    },
}


def find_date_column(df):
    for column in df.columns:
        if column.strip().lower() in {
            "date", "datetime", "timestamp"
        }:
            return column
    raise ValueError(
        f"No date column found. Available columns: {list(df.columns)}"
    )


def load_data():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    raw = pd.read_csv(RAW_DATA_PATH)

    required = {
        "date",
        "predicted_pattern",
        "confidence",
        "forward_return",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV is missing columns: {sorted(missing)}"
        )

    raw_date_col = find_date_column(raw)

    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="coerce"
    ).dt.normalize()

    raw["_date"] = pd.to_datetime(
        raw[raw_date_col], errors="coerce"
    ).dt.normalize()

    predictions["confidence"] = pd.to_numeric(
        predictions["confidence"], errors="coerce"
    )
    predictions["forward_return"] = pd.to_numeric(
        predictions["forward_return"], errors="coerce"
    )

    predictions = predictions.dropna(
        subset=["date", "confidence", "forward_return"]
    ).copy()

    raw = raw.dropna(subset=["_date"]).reset_index(drop=True)

    # Map each signal date to its row in the original chronological data.
    date_to_index = {
        date: index for index, date in enumerate(raw["_date"])
    }
    predictions["signal_index"] = predictions["date"].map(date_to_index)

    unmatched = predictions["signal_index"].isna()
    if unmatched.any():
        examples = predictions.loc[unmatched, "date"].head().tolist()
        raise ValueError(
            "Some prediction dates do not match raw-data dates. "
            f"Examples: {examples}"
        )

    predictions["signal_index"] = predictions[
        "signal_index"
    ].astype(int)

    if "fold" not in predictions.columns:
        predictions["fold"] = -1

    return predictions.sort_values(
        ["signal_index", "fold"]
    ).reset_index(drop=True), raw


def summarize(returns):
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "mean_net_return_pct": np.nan,
            "median_net_return_pct": np.nan,
            "compounded_return_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "profit_factor": np.nan,
        }

    equity = np.concatenate(
        [[1.0], np.cumprod(1.0 + returns)]
    )
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0

    winners = returns[returns > 0].sum()
    losers = -returns[returns < 0].sum()

    return {
        "trades": len(returns),
        "win_rate": (returns > 0).mean() * 100,
        "mean_net_return_pct": returns.mean() * 100,
        "median_net_return_pct": np.median(returns) * 100,
        "compounded_return_pct": (equity[-1] - 1) * 100,
        "max_drawdown_pct": drawdown.min() * 100,
        "profit_factor": (
            winners / losers if losers > 0 else np.nan
        ),
    }


def select_non_overlapping_trades(
    df, raw, strategy_name, directions, threshold
):
    cost = ROUND_TRIP_COST_BPS / 10_000.0
    trades = []
    last_exit_index = -1

    for row in df.itertuples(index=False):
        pattern = row.predicted_pattern
        confidence = row.confidence
        signal_index = row.signal_index

        direction = directions.get(pattern, 0)
        if direction == 0 or confidence < threshold:
            continue

        exit_index = signal_index + HORIZON

        # Keep each trade's exit within the available source data.
        if exit_index >= len(raw):
            continue

        # No overlapping positions within this strategy.
        if signal_index <= last_exit_index:
            continue

        market_return = row.forward_return
        strategy_gross_return = direction * market_return
        strategy_net_return = strategy_gross_return - cost
        long_benchmark_net_return = market_return - cost

        trades.append({
            "strategy": strategy_name,
            "threshold": threshold,
            "fold": row.fold,
            "signal_date": row.date,
            "exit_date": raw.loc[exit_index, "_date"],
            "predicted_pattern": pattern,
            "confidence": confidence,
            "direction": direction,
            "market_forward_return": market_return,
            "strategy_gross_return": strategy_gross_return,
            "strategy_net_return": strategy_net_return,
            "long_benchmark_net_return": long_benchmark_net_return,
        })

        last_exit_index = exit_index

    return pd.DataFrame(trades)


def main():
    predictions, raw = load_data()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_trades = []
    summaries = []

    print(f"Prediction rows: {len(predictions)}")
    print(f"Raw data rows: {len(raw)}")
    print(f"Holding horizon: {HORIZON} rows")
    print(f"Assumed round-trip cost: {ROUND_TRIP_COST_BPS:.1f} bps")
    print("\nEXPLORATORY RESULTS ONLY — previously inspected test folds.\n")

    for strategy_name, directions in STRATEGIES.items():
        for threshold in THRESHOLDS:
            trades = select_non_overlapping_trades(
                predictions,
                raw,
                strategy_name,
                directions,
                threshold,
            )

            if trades.empty:
                strategy_stats = summarize([])
                benchmark_stats = summarize([])
            else:
                strategy_stats = summarize(
                    trades["strategy_net_return"]
                )
                benchmark_stats = summarize(
                    trades["long_benchmark_net_return"]
                )
                all_trades.append(trades)

            summary = {
                "strategy": strategy_name,
                "threshold": threshold,
                **strategy_stats,
                "benchmark_trades": benchmark_stats["trades"],
                "benchmark_compounded_return_pct":
                    benchmark_stats["compounded_return_pct"],
                "benchmark_mean_net_return_pct":
                    benchmark_stats["mean_net_return_pct"],
            }
            summaries.append(summary)

            print(
                f"{strategy_name:15s} "
                f"threshold={threshold:.0%} "
                f"trades={strategy_stats['trades']:4d} "
                f"win_rate={strategy_stats['win_rate']:.2f}% "
                f"mean_net={strategy_stats['mean_net_return_pct']:.4f}% "
                f"compound={strategy_stats['compounded_return_pct']:.2f}% "
                f"max_dd={strategy_stats['max_drawdown_pct']:.2f}% "
                f"PF={strategy_stats['profit_factor']:.3f}"
            )

    summary_df = pd.DataFrame(summaries)
    summary_path = REPORT_DIR / "strategy_research_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        trades_df = pd.DataFrame()

    trades_path = REPORT_DIR / "strategy_research_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    print("\nSaved:")
    print(f"- {summary_path.relative_to(ROOT)}")
    print(f"- {trades_path.relative_to(ROOT)}")
    print(
        "\nReminder: this analysis uses previously inspected test folds. "
        "Do not select a final strategy from these results."
    )


if __name__ == "__main__":
    main()
