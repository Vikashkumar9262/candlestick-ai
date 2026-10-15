
import numpy as np
import pandas as pd

FEATURES = [
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "candle_direction",
    "close_return",
    "range_return",
    "relative_volume",
]

CLASSES = [
    "none",
    "doji",
    "hammer",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",
]

LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    candle_range = (h - l).clip(lower=1e-8)
    body = c - o

    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l

    features = pd.DataFrame(index=df.index)
    features["body_ratio"] = body.abs() / candle_range
    features["upper_wick_ratio"] = upper / candle_range
    features["lower_wick_ratio"] = lower / candle_range
    features["candle_direction"] = np.sign(body)
    features["close_return"] = c.pct_change()
    features["range_return"] = candle_range / c.shift(1)

    if "Volume" in df.columns:
        volume = df["Volume"].astype(float)
        # Use only current and past volume.
        median_volume = volume.rolling(20, min_periods=5).median()
        features["relative_volume"] = (
            volume / median_volume.replace(0, np.nan)
        ).clip(upper=10)
    else:
        features["relative_volume"] = np.nan

    return features.replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame) -> pd.Series:
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    candle_range = (h - l).clip(lower=1e-8)
    body = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l

    labels = pd.Series("none", index=df.index, dtype="object")

    # Precedence matters: the first matching rule wins.
    labels[body / candle_range <= 0.10] = "doji"

    hammer = (
        (lower >= 2 * body)
        & (upper <= 0.5 * body)
        & (body / candle_range <= 0.40)
    )
    labels[hammer] = "hammer"

    shooting_star = (
        (upper >= 2 * body)
        & (lower <= 0.5 * body)
        & (body / candle_range <= 0.40)
    )
    labels[shooting_star] = "shooting_star"

    prev_open = o.shift(1)
    prev_close = c.shift(1)

    bullish_engulfing = (
        (prev_close < prev_open)
        & (c > o)
        & (o <= prev_close)
        & (c >= prev_open)
    )
    bearish_engulfing = (
        (prev_close > prev_open)
        & (c < o)
        & (o >= prev_close)
        & (c <= prev_open)
    )

    labels[bullish_engulfing] = "bullish_engulfing"
    labels[bearish_engulfing] = "bearish_engulfing"

    # The first row has no previous candle.
    labels.iloc[0] = "none"

    return labels


def make_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    window: int = 30,
):
    """Each window ends at its label's candle."""
    X, y, target_indices = [], [], []

    for i in range(window - 1, len(features)):
        window_data = features[i - window + 1:i + 1]

        if not np.isfinite(window_data).all():
            continue

        X.append(window_data.T)  # channels x time
        y.append(labels[i])
        target_indices.append(i)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(target_indices, dtype=np.int64),
    )
