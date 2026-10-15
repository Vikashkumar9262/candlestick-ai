# Frozen Strategy Specification — v1

Status: FROZEN BEFORE FINAL HOLDOUT EVALUATION

## Model
- Model family: CNN candlestick-pattern classifier
- Features, labels, architecture, and preprocessing: keep current implementation unchanged
- Training protocol: chronological; fit preprocessing on training data only
- Final holdout: last 10% of observations, reserved and not inspected

## Strategy
- Strategy: bullish_only
- Confidence threshold: 0.70
- Holding period: 5 trading sessions
- Signal timing: after the signal candle closes
- Entry: next session Open
- Exit: Close on the fifth session after the signal
- Position direction: long only
- Overlapping trades: prohibited within this strategy
- Transaction cost: 10 bps round trip per completed trade

## Evaluation rules
- No threshold or model tuning using final-holdout results
- Include all eligible signals, following the fixed strategy rules
- Report net returns, trade count, win rate, profit factor,
  maximum drawdown, and benchmark comparison
- Benchmark: continuous long exposure over the same holdout window
- Exclude dividends and financing costs; disclose this limitation
- Do not interpret classification accuracy as proof of profitability

## Important limitations
- The 70% threshold and 10 bps cost are research assumptions,
  not proven optimal or guaranteed realistic
- Historical results do not guarantee future performance
