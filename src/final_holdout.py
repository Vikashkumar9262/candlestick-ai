
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import (
    FEATURES,
    CLASSES,
    LABEL_TO_ID,
    make_features,
    make_labels,
    make_sequences,
)
from src.model import CandlestickCNN


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "nifty50.csv"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "final_holdout"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Frozen strategy configuration
WINDOW = 30
HORIZON = 5
HOLDOUT_FRACTION = 0.10
VALIDATION_FRACTION = 0.10
CONFIDENCE_THRESHOLD = 0.70
ROUND_TRIP_COST_BPS = 10

# Existing CNN training settings
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 6
SEED = 42
WEIGHT_DECAY = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def clean_features(df):
    features = make_features(df)

    if "Volume" not in df.columns:
        features["relative_volume"] = 1.0
    else:
        features["relative_volume"] = (
            features["relative_volume"].fillna(1.0)
        )

    return (
        features.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def train_model(X_train, y_train, X_val, y_val):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    present_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train,
    )
    class_weights = np.ones(len(CLASSES), dtype=np.float32)
    class_weights[present_classes] = weights

    model = CandlestickCNN(
        num_features=len(FEATURES),
        num_classes=len(CLASSES),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                total_loss += criterion(model(xb), yb).item() * len(xb)

        val_loss = total_loss / len(val_ds)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"validation_loss={val_loss:.5f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss


def predict_probabilities(model, X):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    results = []

    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            results.append(
                torch.softmax(logits, dim=1).cpu().numpy()
            )

    return np.concatenate(results, axis=0)


def max_drawdown(trade_returns):
    equity = np.cumprod(1.0 + np.asarray(trade_returns, dtype=float))
    equity = np.insert(equity, 0, 1.0)
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1.0))


def profit_factor(returns):
    returns = np.asarray(returns, dtype=float)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else np.nan
    return float(gains / losses)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates("Date", keep="first")
    df = df.set_index("Date")

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index()

    n = len(df)
    holdout_start = int(n * (1.0 - HOLDOUT_FRACTION))
    train_end = int(holdout_start * (1.0 - VALIDATION_FRACTION))
    val_start = train_end + HORIZON
    val_end = holdout_start

    if train_end <= WINDOW or val_start >= val_end:
        raise ValueError("Not enough rows for chronological splits.")

    print("Device:", device)
    print("Total rows:", n)
    print("Training rows:", 0, "to", train_end - 1)
    print("Validation rows:", val_start, "to", val_end - 1)
    print("Holdout rows:", holdout_start, "to", n - 1)
    print("Training ends:", df.index[train_end - 1].date())
    print("Validation:", df.index[val_start].date(),
          "to", df.index[val_end - 1].date())
    print("Holdout starts:", df.index[holdout_start].date())
    print("Holdout ends:", df.index[-1].date())

    features = clean_features(df)
    labels_series = make_labels(df)
    labels = labels_series.map(LABEL_TO_ID).to_numpy()

    # Fit preprocessing using training rows only.
    scaler = StandardScaler()
    scaler.fit(features.iloc[:train_end][FEATURES])
    scaled = scaler.transform(features[FEATURES])

    X, y, target_idx = make_sequences(scaled, labels, WINDOW)

    train_mask = target_idx < train_end
    val_mask = (
        (target_idx >= val_start)
        & (target_idx < val_end)
    )

    # Only evaluate signals whose five-session exit is inside holdout.
    test_mask = (
        (target_idx >= holdout_start)
        & (target_idx < n - HORIZON)
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_idx = target_idx[test_mask]

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError("A split contains no sequences.")

    print(
        "Sequence counts — train/validation/holdout:",
        len(X_train), len(X_val), len(X_test)
    )

    model, best_epoch, best_val_loss = train_model(
        X_train, y_train, X_val, y_val
    )

    # Save reproducible artifacts.
    torch.save(
        model.state_dict(),
        MODEL_DIR / "candlestick_cnn.pt",
    )
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    metadata = {
        "seed": SEED,
        "window": WINDOW,
        "horizon": HORIZON,
        "features": FEATURES,
        "classes": CLASSES,
        "train_end_row_exclusive": train_end,
        "validation_start_row": val_start,
        "validation_end_row_exclusive": val_end,
        "holdout_start_row": holdout_start,
        "holdout_start_date": str(df.index[holdout_start].date()),
        "holdout_end_date": str(df.index[-1].date()),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    probabilities = predict_probabilities(model, X_test)
    predicted_ids = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    holdout = pd.DataFrame({
        "row_index": test_idx,
        "date": df.index[test_idx],
        "actual_pattern": [CLASSES[i] for i in y_test],
        "predicted_pattern": [CLASSES[i] for i in predicted_ids],
        "confidence": confidence,
    })

    # Compute returns from next Open to Close five rows after signal.
    opens = df["Open"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    dates = df.index

    gross_returns = []
    entry_dates = []
    exit_dates = []

    for row_idx in test_idx:
        entry_idx = row_idx + 1
        exit_idx = row_idx + HORIZON
        gross_returns.append(opens[entry_idx] and (
            closes[exit_idx] / opens[entry_idx] - 1.0
        ))
        entry_dates.append(dates[entry_idx])
        exit_dates.append(dates[exit_idx])

    holdout["entry_date"] = entry_dates
    holdout["exit_date"] = exit_dates
    holdout["market_forward_return"] = gross_returns

    holdout.to_csv(
        REPORT_DIR / "final_holdout_predictions.csv", index=False
    )

    accuracy = accuracy_score(y_test, predicted_ids)
    balanced = balanced_accuracy_score(y_test, predicted_ids)
    macro_f1 = f1_score(
        y_test,
        predicted_ids,
        labels=np.arange(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    # Fixed bullish-only strategy: predicted hammer or bullish engulfing.
    bullish_classes = {"hammer", "bullish_engulfing"}
    cost = ROUND_TRIP_COST_BPS / 10_000
    trades = []
    last_exit_idx = -1

    for row in holdout.itertuples(index=False):
        signal_idx = int(row.row_index)

        if row.predicted_pattern not in bullish_classes:
            continue
        if row.confidence < CONFIDENCE_THRESHOLD:
            continue
        if signal_idx <= last_exit_idx:
            continue

        gross = float(row.market_forward_return)
        net = gross - cost

        trades.append({
            "signal_date": row.date,
            "entry_date": row.entry_date,
            "exit_date": row.exit_date,
            "predicted_pattern": row.predicted_pattern,
            "confidence": float(row.confidence),
            "gross_return": gross,
            "net_return": net,
        })
        last_exit_idx = signal_idx + HORIZON

    trade_df = pd.DataFrame(trades)
    trade_df.to_csv(
        REPORT_DIR / "final_holdout_trades.csv", index=False
    )

    net_returns = (
        trade_df["net_return"].to_numpy(dtype=float)
        if not trade_df.empty else np.array([], dtype=float)
    )
    strategy_return = (
        float(np.prod(1.0 + net_returns) - 1.0)
        if len(net_returns) else 0.0
    )

    win_rate = (
        float(np.mean(net_returns > 0))
        if len(net_returns) else np.nan
    )

    # Continuous long benchmark over the holdout window.
    # Enter at the first holdout session's Open and exit at its final Close.
    benchmark_gross = (
        float(closes[-1] / opens[holdout_start] - 1.0)
    )
    benchmark_net = benchmark_gross - cost

    summary = {
        "holdout_start": str(df.index[holdout_start].date()),
        "holdout_end": str(df.index[-1].date()),
        "holdout_rows": n - holdout_start,
        "eligible_signal_rows": len(holdout),
        "classification_accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "macro_f1": float(macro_f1),
        "strategy": "bullish_only",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "horizon_sessions": HORIZON,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "trade_count": len(net_returns),
        "strategy_compounded_net_return_pct": strategy_return * 100,
        "win_rate_pct": win_rate * 100 if np.isfinite(win_rate) else None,
        "profit_factor": profit_factor(net_returns),
        "max_drawdown_pct": max_drawdown(net_returns) * 100,
        "benchmark_gross_return_pct": benchmark_gross * 100,
        "benchmark_net_return_pct": benchmark_net * 100,
        "excess_return_vs_benchmark_pp": (
            strategy_return - benchmark_net
        ) * 100,
        "benchmark_note": (
            "Continuous long from first holdout Open to final Close; "
            "one illustrative round-trip cost; excludes dividends "
            "and financing costs."
        ),
    }

    (REPORT_DIR / "final_holdout_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nFINAL HOLDOUT RESULTS")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print("  reports/final_holdout_predictions.csv")
    print("  reports/final_holdout_trades.csv")
    print("  reports/final_holdout_summary.json")
    print("  models/final_holdout/candlestick_cnn.pt")
    print("  models/final_holdout/scaler.joblib")
    print("  models/final_holdout/metadata.json")


if __name__ == "__main__":
    main()
