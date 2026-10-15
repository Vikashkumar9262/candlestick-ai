from pathlib import Path

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


DATA_PATH = Path("data/raw/nifty50.csv")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models/walk_forward")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 30
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 6
HORIZON = 5
SEED = 42
# Fold 1: train 0-30%, validate 30-40%, test 40-50%.
# Each later fold expands training by 10 percentage points.
INITIAL_TRAIN_FRACTION = 0.30
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
STEP_FRACTION = 0.10
FINAL_HOLDOUT_FRACTION = 0.10
PURGE_ROWS = HORIZON

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_folds(n_rows: int) -> list[dict]:
    """Build expanding chronological folds, reserving the final 10%."""
    holdout_start = int(n_rows * (1.0 - FINAL_HOLDOUT_FRACTION))
    folds = []
    train_fraction = INITIAL_TRAIN_FRACTION
    fold_number = 1

    while True:
        train_end = int(n_rows * train_fraction)
        val_start = train_end + PURGE_ROWS
        val_end = int(
            n_rows * (train_fraction + VALIDATION_FRACTION)
        )
        test_start = val_end + PURGE_ROWS
        test_end = int(
            n_rows
            * (
                train_fraction
                + VALIDATION_FRACTION
                + TEST_FRACTION
            )
        )
        if test_end > holdout_start or test_end <= test_start:
            break

        folds.append({
            "fold": fold_number,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        fold_number += 1
        train_fraction += STEP_FRACTION

    return folds

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
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
def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_number: int,
) -> CandlestickCNN:
    torch.manual_seed(SEED + fold_number)
    np.random.seed(SEED + fold_number)

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
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss_sum += criterion(model(xb), yb).item() * len(xb)

        train_loss = train_loss_sum / max(len(train_ds), 1)
        val_loss = val_loss_sum / max(len(val_ds), 1)
        print(
            f"  Fold {fold_number} | Epoch {epoch + 1:02d} "
            f"train={train_loss:.4f} val={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"  Fold {fold_number}: early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        model.state_dict(),
        MODEL_DIR / f"candlestick_cnn_fold_{fold_number}.pt",
    )
    return model
def predict(
    model: CandlestickCNN,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=128,
        shuffle=False,
    )
    model.eval()
    all_probabilities = []

    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            all_probabilities.append(
                torch.softmax(logits, dim=1).cpu().numpy()
            )

    probabilities = np.concatenate(all_probabilities, axis=0)
    return probabilities.argmax(axis=1), probabilities

def main():
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    features = clean_features(df)
    labels = make_labels(df).map(LABEL_TO_ID).to_numpy()
    folds = make_folds(len(df))

    if not folds:
        raise ValueError("No walk-forward folds generated; check dataset size.")

    print("Device:", device)
    print("Rows:", len(df))
    print("Fold count:", len(folds))
    print("Final untouched holdout starts at row:",
          int(len(df) * (1.0 - FINAL_HOLDOUT_FRACTION)))

    fold_metrics = []
    all_predictions = []
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    for fold in folds:
        number = fold["fold"]
        train_end = fold["train_end"]
        val_start, val_end = fold["val_start"], fold["val_end"]
        test_start, test_end = fold["test_start"], fold["test_end"]

        print(f"\n=== Fold {number} ===")
        print(
            f"Train: 0..{train_end - 1} "
            f"({df.index[0].date()} to {df.index[train_end - 1].date()})"
        )
        print(
            f"Validation: {val_start}..{val_end - 1} "
            f"({df.index[val_start].date()} to {df.index[val_end - 1].date()})"
        )
        print(
            f"Test: {test_start}..{test_end - 1} "
            f"({df.index[test_start].date()} to {df.index[test_end - 1].date()})"
        )

        # Fit preprocessing on this fold's training rows only.
        scaler = StandardScaler()
        scaler.fit(features.iloc[:train_end][FEATURES])
        scaled = scaler.transform(features[FEATURES])

        X, y, target_idx = make_sequences(scaled, labels, WINDOW)
        train_mask = target_idx < train_end
        val_mask = (target_idx >= val_start) & (target_idx < val_end)

        # Keep forward-return outcomes fully inside the test window.
        test_mask = (
            (target_idx >= test_start)
            & (target_idx < test_end - HORIZON)
        )

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_idx = target_idx[test_mask]

        if min(len(X_train), len(X_val), len(X_test)) == 0:
            print("Skipping fold because a split has no sequences.")
            continue

        
        print(
            f"Sequences train/validation/test: "
            f"{len(X_train)}/{len(X_val)}/{len(X_test)}"
        )
        

        model = train_one_fold(
            X_train, y_train, X_val, y_val, number
        )
        joblib.dump(
            scaler, MODEL_DIR / f"scaler_fold_{number}.joblib"
        )

        predictions, probabilities = predict(model, X_test)
        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=np.arange(len(CLASSES)),
            average="macro",
            zero_division=0,
        )

        fold_metrics.append({
            "fold": number,
            "train_end_date": df.index[train_end - 1],
            "validation_start_date": df.index[val_start],
            "validation_end_date": df.index[val_end - 1],
            "test_start_date": df.index[test_start],
            "test_end_date": df.index[test_end - 1],
            "train_sequences": len(X_train),
            "validation_sequences": len(X_val),
            "test_sequences": len(X_test),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        })

        for j, row_idx in enumerate(test_idx):
            entry_idx = row_idx + 1
            exit_idx = row_idx + HORIZON
            entry = open_[entry_idx]
            exit_price = close[exit_idx]
            forward_return = (
                exit_price / entry - 1.0
                if np.isfinite(entry) and entry > 0
                and np.isfinite(exit_price)
                else np.nan
            )
            all_predictions.append({
                "fold": number,
                "date": df.index[row_idx],
                "actual_pattern": CLASSES[y_test[j]],
                "predicted_pattern": CLASSES[predictions[j]],
                "confidence": float(probabilities[j].max()),
                "forward_return": forward_return,
            })

        print(
            f"Fold {number} metrics: accuracy={accuracy:.4f}, "
            f"balanced_accuracy={balanced_accuracy:.4f}, "
            f"macro_f1={macro_f1:.4f}"
        )

    metrics_df = pd.DataFrame(fold_metrics)
    predictions_df = pd.DataFrame(all_predictions)

    metrics_path = REPORT_DIR / "walk_forward_metrics.csv"
    predictions_path = REPORT_DIR / "walk_forward_predictions.csv"
    pattern_path = REPORT_DIR / "walk_forward_pattern_outcomes.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    if not predictions_df.empty:
        pattern_summary = predictions_df.groupby("predicted_pattern").agg(
            samples=("forward_return", "count"),
            mean_forward_return=("forward_return", "mean"),
            median_forward_return=("forward_return", "median"),
            positive_return_rate=(
                "forward_return", lambda s: (s > 0).mean()
            ),
        )
        pattern_summary.to_csv(pattern_path)

    print("\n=== Walk-forward summary ===")
    if not metrics_df.empty:
        print(metrics_df[
            ["fold", "accuracy", "balanced_accuracy", "macro_f1"]
        ].round(4).to_string(index=False))
        print("\nMean across folds:")
        print(
            metrics_df[
                ["accuracy", "balanced_accuracy", "macro_f1"]
            ].mean().round(4).to_string()
        )

    print("\nSaved:")
    print("-", metrics_path)
    print("-", predictions_path)
    if not predictions_df.empty:
        print("-", pattern_path)
    print(
        "\nNote: the final 10% of rows is not used in these folds. "
        "Do not use these fold results to tune repeatedly and then report "
        "them as an untouched final test."
    )


if __name__ == "__main__":
    main()
